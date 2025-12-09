'''
实验名称：字符识别（OCR）
功能：
1. 修复：将模型输出的负数 (LogProb) 转换为标准概率 (0~1)。
2. 拦截：置信度低于 50% (0.5) 的结果，强制显示“字典未收录”。
3. 效果：对于“耄耋”，系统会识别出“耄”，但因为“耋”被识别为低置信度的“奈”，整词将提示未收录（或者您可以选择跳过该字）。
'''

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from media.sensor import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aicube
import random
import gc
import sys
import math # 【新增】引入 math 库用于计算 exp

# ================= 1. 检测类 =================
class OCRDetectionApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,mask_threshold=0.5,box_threshold=0.2,rgb888p_size=[224,224],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.mask_threshold=mask_threshold
        self.box_threshold=box_threshold
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right=self.get_padding_param()
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [0,0,0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            hwc_array=self.chw2hwc(self.cur_img)
            det_boxes = aicube.ocr_post_process(results[0][:,:,:,0].reshape(-1), hwc_array.reshape(-1),self.model_input_size,self.rgb888p_size, self.mask_threshold, self.box_threshold)
            return det_boxes

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * input_width)
        new_h = (int)(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return  top, bottom, left, right

    def chw2hwc(self,features):
        ori_shape = (features.shape[0], features.shape[1], features.shape[2])
        c_hw_ = features.reshape((ori_shape[0], ori_shape[1] * ori_shape[2]))
        hw_c_ = c_hw_.transpose()
        new_array = hw_c_.copy()
        hwc_array = new_array.reshape((ori_shape[1], ori_shape[2], ori_shape[0]))
        del c_hw_
        del hw_c_
        del new_array
        return hwc_array

# ================= 2. 识别类 =================
class OCRRecognitionApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,dict_path,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.dict_path=dict_path
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.dict_word={}
        self.read_dict()
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.RGB_packed,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

        # 【关键设置】真实概率阈值：50%
        # 只有超过 50% 把握的字才显示，否则认为未收录
        self.conf_threshold = 0.5

    def config_preprocess(self,input_image_size=None,input_np=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right=self.get_padding_param(ai2d_input_size,self.model_input_size)
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [0,0,0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([input_np.shape[0],input_np.shape[1],input_np.shape[2],input_np.shape[3]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            preds = np.argmax(results[0], axis=2).reshape((-1))
            probs = np.max(results[0], axis=2).reshape((-1))

            num_classes = results[0].shape[2]
            blank_index = num_classes - 1

            output_txt = ""
            found_suspicious_char = False # 标记是否发现了可疑的字

            # --- 调试打印专用代码 ---
            # unique_preds = []
            # for p in preds:
            #     if p != blank_index and (not unique_preds or p != unique_preds[-1]):
            #         unique_preds.append(p)
            # if len(unique_preds) > 0:
            #     print("\n--- Frame Data ---")

            for i in range(len(preds)):
                if preds[i] == blank_index: continue
                if i > 0 and preds[i - 1] == preds[i]: continue

                idx = preds[i]
                log_prob = probs[i] # 这是负数，例如 -0.9

                # 【核心修复】：将 LogProb 转换为 RealProb (0~1)
                real_prob = math.exp(log_prob)

                char_str = ""
                if idx in self.dict_word:
                    char_str = self.dict_word[idx]
                else:
                    char_str = "?"
                    found_suspicious_char = True

                # print(f"Char: {char_str} | Prob: {real_prob:.2%}") # 打印百分比

                # 【核心判断】：如果任何一个字的真实概率低于 50%
                if real_prob < self.conf_threshold:
                    found_suspicious_char = True
                    # print(f"  -> Low confidence! < {self.conf_threshold}")

                output_txt += char_str

            # 最终决策：
            # 如果这一串字里，有一个字是“索引越界”或者“置信度低”
            # 那么整个词显示为“字典未收录”
            if found_suspicious_char:
                return "字典未收录"

            if output_txt == "":
                return "..."

            return output_txt

    def get_padding_param(self,src_size,dst_size):
        dst_w = dst_size[0]
        dst_h = dst_size[1]
        input_width = src_size[0]
        input_high = src_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * input_width)
        new_h = (int)(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return  top, bottom, left, right

    def read_dict(self):
        self.dict_word = {}
        if self.dict_path!="":
            try:
                with open(self.dict_path, 'r') as file:
                    line_list = file.readlines()
                for num, char in enumerate(line_list):
                    self.dict_word[num] = char.strip()
                print(f"字典加载成功，共 {len(self.dict_word)} 个字符")
            except Exception as e:
                print(f"Error reading dict: {e}")

# ================= 3. 总控类 =================
class OCRDetRec:
    def __init__(self,ocr_det_kmodel,ocr_rec_kmodel,det_input_size,rec_input_size,dict_path,mask_threshold=0.25,box_threshold=0.3,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0, interval=3, min_text_height=15):
        self.ocr_det_kmodel=ocr_det_kmodel
        self.ocr_rec_kmodel=ocr_rec_kmodel
        self.det_input_size=det_input_size
        self.rec_input_size=rec_input_size
        self.dict_path=dict_path
        self.mask_threshold=mask_threshold
        self.box_threshold=box_threshold
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.ocr_det=OCRDetectionApp(self.ocr_det_kmodel,model_input_size=self.det_input_size,mask_threshold=self.mask_threshold,box_threshold=self.box_threshold,rgb888p_size=self.rgb888p_size,display_size=self.display_size,debug_mode=0)
        self.ocr_rec=OCRRecognitionApp(self.ocr_rec_kmodel,model_input_size=self.rec_input_size,dict_path=self.dict_path,rgb888p_size=self.rgb888p_size,display_size=self.display_size)
        self.ocr_det.config_preprocess()

        self.interval = interval
        self.min_text_height = min_text_height
        self.frame_counter = 0
        self.cache_ocr_res = []

    def run(self,input_np):
        self.frame_counter += 1

        det_res_raw = self.ocr_det.run(input_np)

        boxes = []
        ocr_res = []
        valid_dets = []

        for det in det_res_raw:
            h = det[0].shape[1]
            w = det[0].shape[2]
            if h < self.min_text_height or w < 8:
                continue
            valid_dets.append(det)

        is_recognition_frame = (self.frame_counter % self.interval == 0)

        if is_recognition_frame:
            current_frame_text = []
            for det in valid_dets:
                self.ocr_rec.config_preprocess(input_image_size=[det[0].shape[2], det[0].shape[1]], input_np=det[0])
                text = self.ocr_rec.run(det[0])
                current_frame_text.append(text)
                boxes.append(det[1])
                gc.collect()

            self.cache_ocr_res = current_frame_text
            ocr_res = current_frame_text
        else:
            for i, det in enumerate(valid_dets):
                boxes.append(det[1])
                if i < len(self.cache_ocr_res):
                    ocr_res.append(self.cache_ocr_res[i])
                else:
                    ocr_res.append("...")

        return boxes, ocr_res

    def draw_result(self,pl,det_res,rec_res):
        pl.osd_img.clear()
        if det_res:
            for j in range(len(det_res)):
                if j >= len(rec_res): continue

                for i in range(4):
                    x1 = det_res[j][(i * 2)] / self.rgb888p_size[0] * self.display_size[0]
                    y1 = det_res[j][(i * 2 + 1)] / self.rgb888p_size[1] * self.display_size[1]
                    x2 = det_res[j][((i + 1) * 2) % 8] / self.rgb888p_size[0] * self.display_size[0]
                    y2 = det_res[j][((i + 1) * 2 + 1) % 8] / self.rgb888p_size[1] * self.display_size[1]
                    pl.osd_img.draw_line((int(x1), int(y1), int(x2), int(y2)), color=(255, 0, 0, 255),thickness=5)

                text = rec_res[j]
                if text and text != "...":
                    pl.osd_img.draw_string_advanced(int(x1),int(y1),32, text, color=(0,0,255))

if __name__=="__main__":

    display="lcd3_5"

    if display=="hdmi":
        display_mode='hdmi'
        display_size=[1920,1080]
    elif display=="lcd3_5":
        display_mode= 'st7701'
        display_size=[800,480]
    elif display=="lcd2_4":
        display_mode= 'st7701'
        display_size=[640,480]

    # Sensor 设置为 VGA (640x480)
    rgb888p_size=[640, 480]

    ocr_det_kmodel_path="/sdcard/examples/kmodel/ocr_det_int16.kmodel"
    ocr_rec_kmodel_path="/sdcard/examples/kmodel/ocr_rec_int16.kmodel"
    dict_path="/sdcard/examples/utils/dict.txt"

    ocr_det_input_size=[640,640]
    ocr_rec_input_size=[512,32]

    mask_threshold=0.5
    box_threshold=0.6

    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    pl.create(Sensor(width=640, height=480))

    # 正常运行模式
    ocr=OCRDetRec(ocr_det_kmodel_path, ocr_rec_kmodel_path,
                  det_input_size=ocr_det_input_size, rec_input_size=ocr_rec_input_size,
                  dict_path=dict_path, mask_threshold=mask_threshold, box_threshold=box_threshold,
                  rgb888p_size=rgb888p_size, display_size=display_size,
                  interval=3,
                  min_text_height=15)

    clock = time.clock()

    while True:
        clock.tick()
        try:
            img=pl.get_frame()
            det_res,rec_res=ocr.run(img)
            ocr.draw_result(pl,det_res,rec_res)
            pl.show_image()
            gc.collect()
            print(f"FPS: {clock.fps()}")
        except Exception as e:
            print(f"Error: {e}")
            gc.collect()
