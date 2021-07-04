import weakref
from oncedet.cv.utils import Registry
from oncedet.cv.runner import ClosureHook
from oncedet.models import build_loss

__all__ = [
    'BATCHPROCESSORS', 'build_batch_processor',
    'BaseBatchProcessor',
    'DefaultBatchProcessor', 'RandomActiveBatchProcessor',
    'DistillBatchProcessor', 'RandomActiveDistillBatchProcessor'
]

BATCHPROCESSORS = Registry('batch_processor')

def build_batch_processor(cfg, default_args=None):
    return build_from_cfg(cfg, BATCHPROCESSORS, default_args=default_args)

class BaseBatchProcessor:

    def __init__(self, runner):
        # NOTE(ljm): To avoid circular reference, BatchProcess and Runner cannot own each other.
        #            This normally does not matter, but will cause memory leak if the
        #            involved objects contain __del__:
        #            See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
        self.runner = weakref.proxy(runner)

    def __call__(self, **kwargs):
        return self.process(**kwargs)

    def process(self, train_mode, **kwargs):
        return eval(f'self.{train_mode}_process')(**kwargs)

    def train_process(self, data_batch):
        raise NotImplementedError

    def val_process(self, data_batch):
        # NOTE(ljm): the model.set_active_subnet(net_config) is registered
        #            as a ClosureHook in the BatchProcessor when necessary.
        return self.runner.model.val_step(data_batch)

    def train_process_default(self, data_batch):
        return self.runner.model.train_step(data_batch)

    def train_process_with_distill(self, data_batch):
        outputs, outputs_teacher = self.runner.model.train_step(data_batch), self.runner.model_teacher.val_step(data_batch)
        for k, distill_loss in zip(self.distill_keys, self.distill_losses):
            student_pred, teacher_pred = outputs.pop(k), outputs_teacher.pop(k)
            outputs[f'distill_loss_{k}'] = distill_loss(student_pred, teacher_pred)
            outputs['loss'] += outputs[f'distill_loss_{k}']
        return outputs

    def init_distill(self, distill_keys, distill_losses):
        self.distill_keys = distill_keys
        self.distill_losses = [build_loss(distill_loss) for distill_loss in distill_losses]

@BATCHPROCESSORS.register_module()
class DefaultBatchProcessor(BaseBatchProcessor):

    def __init__(**kwargs):
        super(DefaultBatchProcessor, self).__init__(**kwargs)

    def train_process(self, data_batch):
        return self.train_process_default(data_batch)

@BATCHPROCESSORS.register_module()
class RandomActiveBatchProcessor(BaseBatchProcessor):

    def __init__(self, val_subnet_id, random_times=1, mean=False, **kwargs):
        super(DefaultBatchProcessor, self).__init__(**kwargs)
        self.random_times = random_times
        # NOTE(ljm): mean=True will affect the lr when random_times changes
        #            mean=False will affect the epoch when random_times changes
        # TODO(ljm): think out a better way to unify and modify them automatically.
        #            Remind: Make it simple, and let users to know and control everything.
        self.mean = mean

        self.runner.register_hook(ClosureHook('before_val_epoch',
                                              lambda runner: runner.model.set_active_subnet(val_subnet_id)))

    def train_process(self, data_batch):
        outputs = self.train_process_sample_once(data_batch, 0)
        for idx in range(1, self.random_times):
            tmp_outputs = self.train_process_sample_once(data_batch, idx)
            for k in outputs.keys():
                outputs[k] += tmp_outputs[k]
        if self.mean:
            for k in outputs.keys():
                outputs[k] /= self.random_times
        return outputs

    def train_process_sample_once(self, data_batch, idx):
        # TODO(ljm): checkout if the random seed reset in the following lines necessary or not
        subnet_seed = int('%d%.3d%.3d' % (self.runner.iter, idx, 0))
        random.seed(subnet_seed)
        self.runner.model.sample_active_subnet()
        return self.train_process_once(data_batch)

    def train_process_once(self, data_batch):
        return self.train_process_default(data_batch)

@BATCHPROCESSORS.register_module()
class DistillBatchProcessor(BaseBatchProcessor):

    def __init__(self, distill_keys, distill_losses=[dict(type='MSELoss')], **kwargs):
        super(DefaultBatchProcessor, self).__init__(**kwargs)
        self.init_distill(distill_keys, distill_losses)

    def train_process(self, data_batch):
        return self.train_process_with_distill(data_batch)

@BATCHPROCESSORS.register_module()
class RandomActiveDistillBatchProcessor(RandomActiveBatchProcessor):

    def __init__(self, distill_keys, distill_losses=[dict(type='MSELoss')], **kwargs):
        super(RandomActiveDistillBatchProcessor, self).__init__(**kwargs)
        self.init_distill(distill_keys, distill_losses)

    def train_process_once(self, data_batch):
        return self.train_process_with_distill(data_batch)
