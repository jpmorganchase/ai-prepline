# ai_prepline  

- [ai\_prepline](#ai_prepline)
  - [What is ai\_prepline?](#what-is-ai_prepline)
    - [What Do Transformations Look Like?](#what-do-transformations-look-like)
    - [Where are ai\_prepline transformations useful?](#where-are-ai_prepline-transformations-useful)
  - [System requirements](#system-requirements)
  - [Installing](#installing)



## What is ai_prepline?

ai_prepline is a data-processing library! 

ai_prepline enables data scientists and engineers to quickly integrate both original and third-party methods into data preprocessing pipelines, 
with a particular focus on unstructured text data. This allows users to rapidly experiment with preprocessing steps and 
observe their impact on metrics without spending excessive time on boilerplate Python code.

To achieve this, we utilize the Pipeline abstraction from scikit-learn. For example, we can define a few preprocessing steps for our data like this:

```python 
concatenate_text_columns = [
        (
            "concatenate_message",
            string_processing.ConcatTextPerWindow(
                group_by_clms=group_by_clms,
                text_column_to_concatenate=clms_to_concatenate,
                concatenated_column_name=names_for_concatenated_clms,
                separator_for_concatenated_text=separator,
                drop_duplicates=drop_duplicates,
            ),
        ),
    ]

    deduplicate_message_template = [
        (
            "deduplicator",
            string_processing.DeduplicateColumnText(
                text_column_to_deduplicate=("message_template" + "_concatenated"),
                deduplicated_column_name="setfit_input",
                separator=separator,
                prepend_statistics=[len],
                filter_by_columns=group_by_clms,
            ),
        )
    ]

    check_against_output_schema = CheckAgainstSchema(
        output_schema, strict=strict_datacheck
    )

pipeline_list = (
        check_input_schema
        + concatenate_text_columns
        + deduplicate_message_template
        + check_against_output_schema
    )
```

We can then construct the pipeline using `sklearn.pipeline.Pipeline(pipeline_list)`. You can view the entire pipeline in the file `ai_prepline.pipeline_generators.logfile_pipelines.py`.

Implementing this within Python allows us to rapidly experiment with pipeline steps and assess their impact on final metrics.

### What Do Transformations Look Like?

ai_prepline operates by composing transformations of raw data. These transformations are designed to inherit from ai_prepline.base.transform.BaseTransform. This is a lightweight abstraction that requires only two methods:

```python
def fit(self, X: pd.DataFrame, y: Optional[list] = None) -> "BaseTransform":
        """
        Fit dataframe

        :param X: dataframe to fit
        :return: noop - return self
        """
        return self

def transform(self, X: pd.DataFrame, y: Optional[list] = None) -> pd.DataFrame:
    """
    Override me - Implement the data transformation

    :param X: dataframe to transform
    :param y: labels, noop, defaults to None
    :return: transformed data
    """
```

### Where are ai_prepline transformations useful?

We found ai_prepline useful when we wanted to experiment with data-transformations on text-data, 
e.g., prior to language-model finetuning. Then we could reuse the same pipeline for inference. 

Because sometimes you may need better scaling than pandas dataframes can afford, we have tested that
a lot of the transformations work as they are with the [modin](https://github.com/modin-project/modin) library.

We have a two examples of pipelines we have used in the past under `ai_prepline/pipeline_generators/`.

## System requirements

- Python `^3.10`
- Poetry 


## Installing

To install ai_prepline, you first need to install poetry. For example, you can do that by

```bash
pip install poetry
```

Then, you can install the package by using 

```bash 
poetry install
```

Poetry will create a virtual environment for you and install all dependencies. 

