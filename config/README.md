# `attrs_config.json`

This JSON file defines the dataset configuration, including:
1. The number of categories in the dataset.
2. The attributes associated with each category.
3. The classes and their names for each attribute.
   
The structure of the JSON file is as follows:

```
{
  "category_name": {
    "attribute_1": [
      "class_1",
      "class_2",
      "class_3"
         ...      
    ],
    "attribute_2": [
      "class_4",
      "class_5",
      "class_6"
         ...
    ]
    .....
  }
  .........
}
```


## Generating the Configuration JSON
The configuration can be generated using the provided script:
```
python create_config.py --csv_path CSV_PATH --parq_path PARQUET_PATH
```
- `--csv_path CSV_PATH` \
Path to the dataset information CSV file.
-  `--parq_path PARQ_PATH` \
Path to the category attributes Parquet file.
- `--out_path OUT_PATH` \
Path to save the output `JSON` file. Defaults to `./attrs_config.json`.

