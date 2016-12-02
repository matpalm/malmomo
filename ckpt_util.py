import datetime
import os
import tensorflow as tf
import time
import sys
import yaml

def add_opts(parser):
  parser.add_argument('--ckpt-dir', type=str, default=None,
                      help="if set save ckpts to this dir")
  parser.add_argument('--ckpt-save-freq', type=int, default=60,
                      help="freq (sec) to save ckpts to agents to reload from")


class AgentCkptLoader(object):
  def __init__(self, sess, ckpt_dir):
    self.sess = sess
    self.last_loaded_ckpt = None
    assert ckpt_dir, "must set --ckpt-dir to load from"
    self.ckpt_dir = ckpt_dir

  def _most_recent_ckpt(self):
    ckpt_info_file = "%s/checkpoint" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      info = yaml.load(open(ckpt_info_file, "r"))
      assert 'model_checkpoint_path' in info
      most_recent_ckpt = info['model_checkpoint_path']
      return "%s/%s" % (self.ckpt_dir, most_recent_ckpt)
    else:
      return None

  def blocking_load_ckpt(self):
    ckpt = None
    while True:
      ckpt = self._most_recent_ckpt()
      if ckpt is not None: break
      print "waiting for ckpt in dir", self.ckpt_dir
      time.sleep(1)
    self.saver = tf.train.Saver(var_list=tf.all_variables())
    self.saver.restore(self.sess, ckpt)
    self.last_loaded_ckpt = ckpt

  def reload_if_new_ckpt(self):
    ckpt = self._most_recent_ckpt()
    assert ckpt is not None
    if ckpt != self.last_loaded_ckpt:
      print "new ckpt [%s] (newer than [%s])" % (ckpt, self.last_loaded_ckpt)
      self.saver.restore(self.sess, ckpt)
      self.last_loaded_ckpt = ckpt


class TrainerCkptSaver(object):
  def __init__(self, sess, ckpt_dir="/tmp", save_freq=60):
    self.sess = sess
    self.saver = tf.train.Saver(var_list=tf.all_variables(),
                                max_to_keep=10,
                                keep_checkpoint_every_n_hours=1)
    self.ckpt_dir = ckpt_dir
    assert ckpt_dir, "must set --ckpt-dir"
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)
    assert save_freq > 0
    self.save_freq = save_freq
    self.load_latest_ckpt_or_init_if_none()

  def load_latest_ckpt_or_init_if_none(self):
    """loads latests ckpt from dir. if there are non run init variables."""
    # if no latest checkpoint init vars and return
    ckpt_info_file = "%s/checkpoint" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      # load latest ckpt
      info = yaml.load(open(ckpt_info_file, "r"))
      assert 'model_checkpoint_path' in info
      most_recent_ckpt = info['model_checkpoint_path']
      sys.stderr.write("loading ckpt %s/%s\n" % (self.ckpt_dir, most_recent_ckpt))
      self.saver.restore(self.sess, self.ckpt_dir+"/"+most_recent_ckpt)
      self.next_scheduled_save_time = time.time() + self.save_freq
    else:
      # no latest ckpts, init and force a save now
      sys.stderr.write("no latest ckpt in %s, just initing vars...\n" % self.ckpt_dir)
      self.sess.run(tf.initialize_all_variables())
      self.force_save()

  def force_save(self):
    """force a save now."""
    dts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_ckpt = "%s/ckpt.%s" % (self.ckpt_dir, dts)
    sys.stderr.write("saving ckpt %s\n" % new_ckpt)
    self.saver.save(self.sess, new_ckpt)
    self.next_scheduled_save_time = time.time() + self.save_freq

  def save_if_required(self):
    """check if save is required based on time and if so, save."""
    if time.time() >= self.next_scheduled_save_time:
      self.force_save()
