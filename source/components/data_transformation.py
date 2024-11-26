import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer object for data preprocessing.
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Pipelines for numerical and categorical data
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder())
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(f"Error in get_data_transformer_object: {str(e)}", sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Performs data transformation on train and test datasets.

        Args:
        - train_path: Path to training dataset CSV
        - test_path: Path to testing dataset CSV

        Returns:
        - Tuple containing transformed train array, test array, and preprocessor file path
        """
        try:
            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Ensure alignment of dimensions
            assert input_feature_train_arr.shape[0] == target_feature_train_df.shape[0], \
                "Mismatch in training features and target row counts"
            assert input_feature_test_arr.shape[0] == target_feature_test_df.shape[0], \
                "Mismatch in testing features and target row counts"

            # Combine transformed features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(f"Error in initiate_data_transformation: {str(e)}", sys)
