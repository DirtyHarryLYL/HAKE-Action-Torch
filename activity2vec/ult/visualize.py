#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
import os
from anycurve import losscurve

class a2v_curve():
    def __init__(self, cfg):
        self.cfg = cfg
        self.curve = self.setup()

    def setup(self):
        loss_curve_dir = os.path.join(self.cfg.LOG_DIR, 'loss_curve_db')
            
        loss_curve = losscurve(db_path=loss_curve_dir, db_name='data', figsize=(20, 12))
        loss_curve.add_key('loss')

        if len(self.cfg.MODEL.MODULE_TRAINED) == 1 and self.cfg.MODEL.MODULE_TRAINED[0] != 'verb':
            loss_curve.add_key('{:s}_map_w_no_interaction'.format(self.cfg.MODEL.MODULE_TRAINED[0]))
            loss_curve.add_key('{:s}_map_wo_no_interaction'.format(self.cfg.MODEL.MODULE_TRAINED[0]))
        else:
            loss_curve.add_key('pasta_map')
            loss_curve.add_key('verb_map')

        loss_curve.set_xlabel('iteration')
        loss_curve.set_ylabel('loss', False)
        loss_curve.set_ylabel('mAP', True)
        loss_curve.daemon(True, self.cfg.TRAIN.SHOW_INTERVAL // self.cfg.TRAIN.DISPLAY_INTERVAL)
        return loss_curve

    def log(self, feed_dict):
        self.curve.log(feed_dict)

    def render(self):
        if self.curve.daemon():
            self.curve.clean()
            self.curve.draw('iteration', 'loss', self.cfg.MODEL_NAME + '_loss')
            self.curve.twin()

            self.curve.clean()
            if len(self.cfg.MODEL.MODULE_TRAINED) == 1 and self.cfg.MODEL.MODULE_TRAINED[0] != 'verb':
                self.curve.draw('iteration', '{:s}_map_w_no_interaction'.format(self.cfg.MODEL.MODULE_TRAINED[0]), self.cfg.MODEL_NAME + '_' + self.cfg.MODEL.MODULE_TRAINED[0] + '_w_nointer')
                self.curve.draw('iteration', '{:s}_map_wo_no_interaction'.format(self.cfg.MODEL.MODULE_TRAINED[0]), self.cfg.MODEL_NAME + '_' + self.cfg.MODEL.MODULE_TRAINED[0] + '_wo_nointer')
            else:
                self.curve.draw('iteration', 'pasta_map', self.cfg.MODEL_NAME + '_pasta')
                self.curve.draw('iteration', 'verb_map', self.cfg.MODEL_NAME + '_verb')
            self.curve.twin()

            self.curve.reset_choice()
            self.curve.legend(inside=False)
            self.curve.synchronize()
            self.curve.render(os.path.join(self.cfg.LOG_DIR, 'curve.png'))