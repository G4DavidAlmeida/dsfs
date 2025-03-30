from scratch.datasets.utils import BaseDirPaths


class SalaryPredictionPaths(BaseDirPaths):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.glassdoor_csv = self._check_file(self.base_dir / 'glassdoor_jobs.csv')
        self.salary_data_cleaned = self._check_file(self.base_dir / 'salary_data_cleaned.csv')
        self.eda_data = self._check_file(self.base_dir / 'eda_data.csv')

class TitanicDatasetPaths(BaseDirPaths):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.tested_csv = self._check_file(self.base_dir / 'tested.csv')
        

class IrisDatasetPaths(BaseDirPaths):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.iris_csv = self._check_file(self.base_dir / 'Iris.csv')
        self.database_sqlite = self._check_file(self.base_dir / 'database.sqlite')

class PredictingHiringDecisionsInRecruitmentDataPaths(BaseDirPaths):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.recruitment_data_csv = self._check_file(self.base_dir / 'recruitment_data.csv')

class SocialNetworkAdsPaths(BaseDirPaths):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.social_network_ads_csv = self._check_file(self.base_dir / 'Social_Network_Ads.csv')