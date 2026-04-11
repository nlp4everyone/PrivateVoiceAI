from app.schema.language import LanguageProperties, LanguageDetail
import pycld2 as cld2


class LanguageDetector:
    """
    A language detector that uses pycld2 to identify text languages.
    
    Provides detection of the primary language and detailed information
    about all detected languages in the given text.
    """
    
    @staticmethod
    def detect(text: str,
               best_effort: bool = True) -> LanguageProperties:
        """
        Detect the language of the given text.
        
        Args:
            text: The text to analyze for language detection
            best_effort: Whether to use best effort detection if uncertain
            
        Returns:
            LanguageProperties containing the detected language information
            including primary language, ISO code, percentage, and detailed results
        """
        # Get the most likely language (ISO 639-1 code)
        _, _, details = cld2.detect(text,
                                    bestEffort = best_effort)
        # Define main language, lang code and its percent
        language, lang_code, percent, _ = details[0]
        # Define lang details
        details = [LanguageDetail(language = str(detail[0]).lower(),
                                  lang_code = detail[1],
                                  percent = detail[2],
                                  score = detail[3]) for detail in details]
        # Return
        return LanguageProperties(language = str(language).lower(),
                                  lang_code = lang_code,
                                  percent = percent,
                                  details = details)

