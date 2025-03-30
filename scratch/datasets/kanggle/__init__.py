import kagglehub

from scratch.datasets.utils import BaseDatasets, BaseDatasetsNames
from .dir_paths import *

class KanggleDatasetNames(BaseDatasetsNames):
    SALARY_PREDICTION = 'thedevastator/jobs-dataset-from-glassdoor'
    TITANIC_DATASET = 'brendan45774/test-file'
    IRIS_DATASET = 'uciml/iris'
    PREDICTING_HIRING_DECISIONS_IN_RECRUITMENT_DATA = 'rabieelkharoua/predicting-hiring-decisions-in-recruitment-data'
    SOCIAL_NETWORK_ADS = 'rakeshrau/social-network-ads'

class KanggleDatasets(BaseDatasets):
    def __init__(self):
        self.kanggle = kagglehub
        
    def _download(self, dataset_name: KanggleDatasetNames):
        return self.kanggle.dataset_download(dataset_name.val_name())
    
    @property
    def salary_prediction(self):
        return SalaryPredictionPaths(self._download(
            KanggleDatasetNames.SALARY_PREDICTION))
    
    @property
    def titanic_dataset(self):
        return TitanicDatasetPaths(self._download(
            KanggleDatasetNames.TITANIC_DATASET))
    
    @property
    def iris_dataset(self):
        return IrisDatasetPaths(self._download(
            KanggleDatasetNames.IRIS_DATASET))
    
    @property
    def predicting_hiring_decisions_in_recruitment_data(self):
        return PredictingHiringDecisionsInRecruitmentDataPaths(self._download(
            KanggleDatasetNames.PREDICTING_HIRING_DECISIONS_IN_RECRUITMENT_DATA))
    
    @property
    def social_network_ads(self):
        return SocialNetworkAdsPaths(self._download(
            KanggleDatasetNames.SOCIAL_NETWORK_ADS))