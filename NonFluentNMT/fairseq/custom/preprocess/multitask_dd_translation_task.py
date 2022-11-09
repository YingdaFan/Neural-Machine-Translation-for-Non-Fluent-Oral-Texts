from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data.dictionary import Dictionary


class Dictionary_SequenceLabel(Dictionary):
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = len(self.symbols)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d


@register_task('multitask_dd_translation_task')
class MultitaskDDTranslationTask(TranslationTask):

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if filename.split(".")[1] == "label":
            return Dictionary_SequenceLabel.load(filename)
        else:
            return Dictionary.load(filename)