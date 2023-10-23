# Adding 4 New Datasets

1. Navigate to the `data` folder. Inside each new dataset folder, place the corresponding files as indicated by the `.gitkeep` file.
2. Once you've added the necessary files, still within the `data` folder, execute the following command to generate the flow data needed for model training as described in our manuscript:
   ```
   python new_dataset_generator.py
   ```
3. After executing the scripts, an `all` folder will appear within each new dataset directory. The name of this folder represents `the timeframe or year` associated with the dataset.
   * For instance, in the **original** datasets, you might observe a `2019` folder. This signifies that only timestamps from the year 2019 are included in that dataset.
   * Conversely, in the **new** datasets, the name `all` indicates that every available timestamp is used, with no filtering applied.
