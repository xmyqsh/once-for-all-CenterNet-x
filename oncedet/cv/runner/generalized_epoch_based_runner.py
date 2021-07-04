from oncedet.cv.runner import RUNNERS, EpochBasedRunner
from oncedet.cv.batch_processor import DefaultBatchProcessor

@RUNNERS.register_module()
class GeneralizedEpochBasedRunner(EpochBasedRunner):
    '''GeneralizedEpochBasedRunner
       Support random active subnet and knowledge distillation currently.
    '''

    def __init__(self, model_teacher=None, *args, **kwargs):
        super(GeneralizedEpochBasedRunner, self).__init__(*args, **kwargs)

        self.model_teacher = model_teacher
        self.batch_processor = self.batch_processor(self) if self.batch_processor is not None else \
                               DefaultBatchProcessor(self)

        self.model_teacher.eval()

    def run_iter(self, *args, **kwargs):
        outputs = self.batch_processor(*args, **kwargs)
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
