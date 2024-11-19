import pandas as pd
import json
import argparse

def get_attributes_name(parq_path: str):
    attrs_name = dict()
    cat_attrs = pd.read_parquet(parq_path)
    for name, attrs in zip(cat_attrs['Category'].tolist(), cat_attrs['Attribute_list'].tolist()):
        attrs_name[name] = attrs.tolist()
    return attrs_name

def create_config(train_df: pd.DataFrame, attrs_name: dict):
    attrs_val_dict = dict()
    for cat_name, attrs in attrs_name.items():
        temp_df = train_df[train_df['Category'] == cat_name]
        attr_meta = dict()
        for num, attr in enumerate(attrs, start=1):
            attr_meta[attr] = sorted(temp_df[f'attr_{num}'].dropna().unique().tolist())
        attrs_val_dict[cat_name] = attr_meta
    return attrs_val_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates Config JSON file contains information of Categories and its attributes and class names of respective attributes"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help="Path to the Dataset Information CSV file."
    )
    parser.add_argument(
        '--parq_path',
        type=str,
        required=True,
        help="Path to the Category Attributes Parquet file."
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default="./attrs_config.json",
        help="Path to save the output JSON file (default: './attrs_config.json')."
    )
    args = parser.parse_args()

    attrs_name = get_attributes_name(args.parq_path)
    df = pd.read_csv(args.csv_path)
    config = create_config(df, attrs_name)
    assert args.out_path.endswith(".json"), "Invalid JSON Path"
    with open(args.out_path, 'w') as j_file:
        json.dump(config, j_file, indent=4)
