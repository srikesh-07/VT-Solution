import torch
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


class VisualTaxonomyF1Metric:
    def __init__(self, label_config: dict, device: str = "cpu"):
        self._init_validators(label_config, device)
        self.device = device

    def _init_validators(self, label_config: dict, device: str):
        self.validators = dict()
        for name, attrs in label_config.items():
            self.validators[name] = list()
            for attr_vals in attrs.values():
                self.validators[name].append(
                    (
                        MulticlassF1Score(
                            num_classes=len(attr_vals), average="micro"
                        ).to(device),
                        MulticlassF1Score(
                            num_classes=len(attr_vals), average="macro"
                        ).to(device),
                    )
                )

    def update(
        self,
        preds: list,
        gts: list,
        cats: list,
        mask: list = None,
        verify: bool = False,
    ):
        if verify:
            for cat in cats[1:]:
                assert cat == cats[0], "Mixing of Different Categories Found.."
        if mask is None:
            mask = [None] * len(gt)
        for (micro_scorer, macro_scorer), pred, gt, m in zip(
            self.validators[cats[0]], preds, gts, mask
        ):
            if verify:
                assert pred.shape[-1] == micro_scorer.num_classes
                assert (
                    getattr(macro_scorer, "num_classes", None) is not None
                    and macro_scorer.num_classes == pred.shape[-1]
                )
            pred, gt = pred.to(self.device), gt.to(self.device)
            if mask is not None:
                m = m.bool().to(self.device)
                pred = pred[m, :]
                gt = gt[m]
                if gt.numel() == 0:
                    continue
            micro_scorer.update(pred, gt)
            macro_scorer.update(pred, gt)

    def calculate_attr_score(self, micro_score: float, macro_score: float):
        if micro_score == 0 and macro_score == 0:
            return 0
        else:
            return (2 * (micro_score * macro_score)) / (micro_score + macro_score)

    def compute(self, classwise_scores: bool = True):
        if classwise_scores:
            scores = {"H.M F1-Score": 0, "Micro F1-Score": 0, "Macro F1-Score": 0}
            attrwise_scores = dict()
            catwise_scores = dict()
        else:
            total_score = 0
        for name, validators in self.validators.items():
            catwise_scores[name] = dict()
            attrwise_scores[name] = {
                "H.M F1-Score": list(),
                "Micro F1-Score": list(),
                "Macro F1-Score": list(),
            }
            attr_scores = 0
            micro_scores = 0
            macro_scores = 0
            for validator in validators:
                micro_validator, macro_validator = validator
                micro_score = micro_validator.compute().item()
                macro_score = macro_validator.compute().item()
                attr_score = self.calculate_attr_score(
                    micro_score=micro_score,
                    macro_score=macro_score,
                )
                attrwise_scores[name]["H.M F1-Score"].append(attr_score)
                attrwise_scores[name]["Micro F1-Score"].append(micro_score)
                attrwise_scores[name]["Macro F1-Score"].append(macro_score)
                attr_scores += attr_score
                micro_scores += micro_score
                macro_scores += macro_score

            inter_score = attr_scores / len(
                validators
            )  # divide by number of attribures
            inter_mac_score = macro_scores / len(validators)
            inter_mic_score = micro_scores / len(validators)
            if classwise_scores:
                scores[f"H.M F1-Score"] += inter_score
                scores[f"Micro F1-Score"] += inter_mic_score
                scores[f"Macro F1-Score"] += inter_mac_score
                catwise_scores[name.title()][f"H.M F1-Score"] = inter_score
                catwise_scores[name.title()][f"Micro F1-Score"] = inter_mic_score
                catwise_scores[name.title()][f"Macro F1-Score"] = inter_mac_score
            else:
                total_score += inter_score
        if classwise_scores:
            scores[f"H.M F1-Score"] /= len(self.validators)
            scores[f"Micro F1-Score"] /= len(self.validators)
            scores[f"Macro F1-Score"] /= len(self.validators)
            return {
                "Overall Scores": scores,
                "Category-Wise Scores": catwise_scores,
                "Attribute-Wise Scores": attrwise_scores,
            }
        else:
            total_score /= len(self.validators)  # number of categories
            return total_score

    def reset(self):
        for validator_list in self.validators.values():
            for micro_valid, macro_valid in validator_list:
                micro_valid.reset()
                macro_valid.reset()
