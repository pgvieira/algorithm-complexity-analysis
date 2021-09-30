from app_analysis import AppAnalysis
from app_extract_value import AppExtractValue


class AppService:
    def __init__(self):
        self.app_analysis = AppAnalysis()
        self.app_extract_value = AppExtractValue()
        pass

    def calculate_complexity(self, analysis_type, filename):
        values_from_file = self.app_extract_value.extract_values(filename)

        if analysis_type == '0':
            return self.app_analysis.calculate_complexity_binomial(values_from_file)
        if analysis_type == '1':
            return self.app_analysis.calculate_complexity_multinomial(values_from_file)
