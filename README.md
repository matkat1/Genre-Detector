# Genre-Detector

## Dataset

https://huggingface.co/datasets/BrightData/Goodreads-Books

The data_goodreads.csv dataset from the link above contains only records that satisfy the following conditions:

1. The summary is longer than 30 characters.
2. The genres field is not NULL, an empty string, or an empty list.

We kept only two columns in data_goodreads.csv: `summary` and `genres` (target).