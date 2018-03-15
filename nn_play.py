# -*- coding:utf-8 -*-
# Created Time: 六 12/30 13:49:21 2017
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import time
import os, glob, shutil
import cv2
import argparse
import tensorflow as tf
from model import JumpModel
from model_fine import JumpModelFine
import random
import sys

# 多尺度搜索
def multi_scale_search(pivot, screen, range=0.3, num=10):
    # 分别获取屏幕截图及player的尺寸
    H, W = screen.shape[:2]
    h, w = pivot.shape[:2]

    found = None
    # 利用np.linspace(0.7,1.3,10)[::-1]生成0.7-1.3之间的10个等差数，-1指定倒序排列
    for scale in np.linspace(1-range, 1+range, num)[::-1]:
        # 对截屏按等差数列进行缩放：
        resized = cv2.resize(screen, (int(W * scale), int(H * scale)))

        # 计算缩放率 rate
        r = W / float(resized.shape[1])
        # 如果缩放后的截屏比player的尺寸还小，停止匹配player
        if resized.shape[0] < h or resized.shape[1] < w:
            break
        # 使用：归一化相关系数匹配法 进行模板匹配和识别
        # 使用的是相关匹配算法，res越大匹配效果越准确
        # res存储的是一幅灰度图片，每个元素表示其附近元素与模板的匹配度
        res = cv2.matchTemplate(resized, pivot, cv2.TM_CCOEFF_NORMED)

        # 通过res >=res.max()来判定本次识别结果是否比之前最好还好，
        # 如果是，则更新player位于截屏中坐标
        loc = np.where(res >= res.max())
        pos_h, pos_w = list(zip(*loc))[0]

        if found is None or res.max() > found[-1]:
            found = (pos_h, pos_w, r, res.max())

    if found is None: return (0,0,0,0,0)
    pos_h, pos_w, r, score = found
    start_h, start_w = int(pos_h * r), int(pos_w * r)
    end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
    return [start_h, start_w, end_h, end_w, score]

class WechatAutoJump(object):
    def __init__(self, phone, sensitivity, serverURL, debug, resource_dir):
        self.phone = phone
        self.sensitivity = sensitivity
        self.debug = debug
        self.resource_dir = resource_dir
        # 初始化已跳跃步数
        self.step = 0
        self.ckpt = os.path.join(self.resource_dir, 'train_logs_coarse/best_model.ckpt-13999')
        self.ckpt_fine = os.path.join(self.resource_dir, 'train_logs_fine/best_model.ckpt-53999')
        self.serverURL = serverURL

        # 加载：player.png，初始化tf.Session()
        self.load_resource()
        if self.phone == 'IOS':
            import wda
            # 连接到手机
            self.client = wda.Client(self.serverURL)
            # 启动应用
            self.s = self.client.session()
        if self.debug:
            if not os.path.exists(self.debug):
                os.mkdir(self.debug)

    def load_resource(self):
        # 加载 小人图片 player.png
        self.player = cv2.imread(os.path.join(self.resource_dir, 'player.png'), 0)

        # network initization
        self.net = JumpModel()
        self.net_fine = JumpModelFine()

        # 定义占位符:
        # 将采集到的大小为1280*720的图像沿x方向上下各截去320*720大小，只保留中心640*720的图像作为训练数据
        self.img = tf.placeholder(tf.float32, [None, 640, 720, 3], name='img')

        self.img_fine = tf.placeholder(tf.float32, [None, 320, 320, 3], name='img_fine')

        # 定义标签:
        self.label = tf.placeholder(tf.float32, [None, 2], name='label')

        self.is_training = tf.placeholder(np.bool, name='is_training')
        self.keep_prob = tf.placeholder(np.float32, name='keep_prob')

        #
        self.pred = self.net.forward(self.img, self.is_training, self.keep_prob)
        self.pred_fine = self.net_fine.forward(self.img_fine, self.is_training, self.keep_prob)

        # 初始化并运行 self.sess
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        all_vars = tf.all_variables()
        var_coarse = [k for k in all_vars if k.name.startswith('coarse')]
        var_fine = [k for k in all_vars if k.name.startswith('fine')]

        self.saver_coarse = tf.train.Saver(var_coarse)
        self.saver_fine = tf.train.Saver(var_fine)
        self.saver_coarse.restore(self.sess, self.ckpt)
        self.saver_fine.restore(self.sess, self.ckpt_fine)

        print('==== successfully restored ====')

    # 获取手机屏幕当前截图, 将截屏缩放成尺寸为：1280*720的图片返回
    def get_current_state(self):
        # 获取当前手机屏截屏，并把图片拉取到程序运行的当前目录
        if self.phone == 'Android':
            os.system('adb shell screencap -p /sdcard/1.png')
            os.system('adb pull /sdcard/1.png state.png')
        elif self.phone == 'IOS':
            self.client.screenshot('state.png')
        if not os.path.exists('state.png'):
            raise NameError('Cannot obtain screenshot from your phone! Please follow the instructions in readme!')

        if self.debug:
            shutil.copyfile('state.png', os.path.join(self.debug, 'state_{:03d}.png'.format(self.step)))

        # 读取这张截图
        state = cv2.imread('state.png')
        # iphone上得到的state的值是：(1334，750,3), 切片取前2个值
        # resolution[0]=y, resolution[1]=x
        # 另外一种赋值方式： rows, columns=state.shape[:2]
        self.resolution = state.shape[:2]

        # 下面要将采集到的图片等比例缩放成尺寸(x,y)：720*1280
        scale = state.shape[1] / 720.  # 计算x轴像素的缩放系数,然后应用到y轴进行缩放
        # 这里 state.shape[0]/scale = 1280.639999，取整后刚好是1280
        state = cv2.resize(state, (720, int(state.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

        # 如果缩放后，state.shape[0]的值还不是1280，要再进一步处理：
        if state.shape[0] > 1280:
            s = (state.shape[0] - 1280) // 2
            state = state[s:(s+1280),:,:]
        elif state.shape[0] < 1280:
            s1 = (1280 - state.shape[0]) // 2
            s2 = (1280 - state.shape[0]) - s1
            pad1 = 255 * np.ones((s1, 720, 3), dtype=np.uint8)
            pad2 = 255 * np.ones((s2, 720, 3), dtype=np.uint8)
            state = np.concatenate((pad1, state, pad2), 0)
        # 后续操作：每张图有判断意义的区域只有屏幕中央位置，截图的上下两部分是没有意义的
        # 后面会从上下各截去320*720大小，只保留中心640*720的图像作为训练数据
        return state

    def get_player_position(self, state):
        # 转换为灰度图片
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # 搜索player的坐标
        pos = multi_scale_search(self.player, state, 0.3, 10)
        h, w = int((pos[0] + 13 * pos[2])/14.), (pos[1] + pos[3])//2
        return np.array([h, w])

    def get_target_position(self, state, player_pos):
        feed_dict = {
            self.img: np.expand_dims(state[320:-320], 0),
            self.is_training: False,
            self.keep_prob: 1.0,
        }
        pred_out = self.sess.run(self.pred, feed_dict=feed_dict)
        pred_out = pred_out[0].astype(int)
        x1 = pred_out[0] - 160
        x2 = pred_out[0] + 160
        y1 = pred_out[1] - 160
        y2 = pred_out[1] + 160
        if y1 < 0:
            y1 = 0
            y2 = 320
        if y2 > state.shape[1]:
            y2 = state.shape[1]
            y1 = y2 - 320
        img_fine_in = state[x1: x2, y1: y2, :]
        feed_dict_fine = {
            self.img_fine: np.expand_dims(img_fine_in, 0),
            self.is_training: False,
            self.keep_prob: 1.0,
        }
        pred_out_fine = self.sess.run(self.pred_fine, feed_dict=feed_dict_fine)
        pred_out_fine = pred_out_fine[0].astype(int)
        out = pred_out_fine + np.array([x1, y1])
        return out

    def get_target_position_fast(self, state, player_pos):
        state_cut = state[:player_pos[0],:,:]
        m1 = (state_cut[:, :, 0] == 245)
        m2 = (state_cut[:, :, 1] == 245)
        m3 = (state_cut[:, :, 2] == 245)
        m = np.uint8(np.float32(m1 * m2 * m3) * 255)
        b1, b2 = cv2.connectedComponents(m)
        for i in range(1, np.max(b2) + 1):
            x, y = np.where(b2 == i)
            if len(x) > 280 and len(x) < 310:
                r_x, r_y = x, y
        h, w = int(r_x.mean()), int(r_y.mean())
        return np.array([h, w])

    def jump(self, player_pos, target_pos):
        distance = np.linalg.norm(player_pos - target_pos)
        press_time = distance * self.sensitivity
        press_time = int(np.rint(press_time))
        press_h, press_w = int(0.82*self.resolution[0]), self.resolution[1]//2
        if self.phone == 'Android':
            cmd = 'adb shell input swipe {} {} {} {} {}'.format(press_w, press_h, press_w, press_h, press_time)
            print(cmd)
            os.system(cmd)
        elif self.phone == 'IOS':
            self.s.tap_hold(press_w, press_h, press_time / 1000.)

    def debugging(self):
        current_state = self.state.copy()
        cv2.circle(current_state, (self.player_pos[1], self.player_pos[0]), 5, (0,255,0), -1)
        cv2.circle(current_state, (self.target_pos[1], self.target_pos[0]), 5, (0,0,255), -1)
        cv2.imwrite(os.path.join(self.debug, 'state_{:03d}_res_h_{}_w_{}.png'.format(self.step, self.target_pos[0], self.target_pos[1])), current_state)

    # Added by yichen
    def personification(self):
        if self.step % 70 == 0:
            next_rest = 18
            rest=True
        elif self.step % 40 == 0:
            next_rest = 13
            rest=True
        elif self.step % 20 == 0:
            next_rest = 11
            rest=True
        elif self.step % 10 == 0:
            next_rest = 8
            rest=True
        else:
            rest=False

        if rest:
            for rest_time in range(next_rest):
                sys.stdout.write('\r程序将在 {}s 后继续' .format(next_rest-rest_time))
                sys.stdout.flush()
                time.sleep(1)
            print('\n继续')

        time.sleep(random.uniform(1.5, 3.0))

        if self.step % 5 == 0:
            self.sensitivity = 2.145
        elif self.step % 7 == 0:
            self.sensitivity = 2.000
        elif self.step % 9 == 0:
            self.sensitivity = 1.985
        elif self.step % 3 == 0:
            self.sensitivity = 1.970

    def play(self):
        # 获取 1280*720大小的屏幕截图
        self.state = self.get_current_state()
        # 计算 player的坐标
        self.player_pos = self.get_player_position(self.state)

        # 计算player要跳到哪个坐标
        if self.phone == 'IOS':
            self.target_pos = self.get_target_position(self.state, self.player_pos)
            print('CNN-search: %04d' % self.step)
        else:
            try:
                self.target_pos = self.get_target_position_fast(self.state, self.player_pos)
                print('fast-search: %04d' % self.step)
            except UnboundLocalError:
                self.target_pos = self.get_target_position(self.state, self.player_pos)
                print('CNN-search: %04d' % self.step)
        if self.debug:
            self.debugging()

        # 触发跳跃动作
        self.jump(self.player_pos, self.target_pos)
        self.step += 1

        time.sleep(1.5)


    def run(self):
        try:
            while True:
                self.play()
        except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone', default='Android', choices=['Android', 'IOS'], type=str, help='mobile phone OS')
    parser.add_argument('--sensitivity', default=2.045, type=float, help='constant for press time')
    parser.add_argument('--serverURL', default='http://localhost:8100', type=str, help='ServerURL for wda Client')
    parser.add_argument('--resource', default='resource', type=str, help='resource dir')
    parser.add_argument('--debug', default=None, type=str, help='debug mode, specify a directory for storing log files.')
    args = parser.parse_args()
    # print(args)

    # 初始化对象
    AI = WechatAutoJump(args.phone, args.sensitivity, args.serverURL, args.debug, args.resource)
    # 调用run()方法->self.play()
    AI.run()
