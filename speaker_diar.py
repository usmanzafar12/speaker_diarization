import argparse
import logging
import matplotlib
import copy
import os
from matplotlib import pyplot as plot
from s4d.utils import *
from s4d.diar import Diar
from s4d import viterbi, segmentation
from s4d.clustering import hac_bic
from sidekit.sidekit_io import init_logging
from s4d.gui.dendrogram import plot_dendrogram
from s4d import scoring
from s4d.model_iv import ModelIV
from sidekit.sidekit_io import *
from sidekit.bosaris import IdMap, Scores
from s4d.clustering.hac_iv import hac_iv
from s4d import scoring
import numpy as np
import traceback
import configparser


class SpeakerDiar(object):
    
    def __init__(self, show, dir, config):

        #Audio filenames and directory stucture
        self.show = show
        self.dir = dir
        self.config = config
        self.cep = None
        self.vad_labels = None
        #Parameters
        self.win_size = 250

        #linear bic segmentation parameters
        #postive value of delta bic indicates we need two separate models for the adjacent segments. 
        #Here we define the threshodls
        self.li_bic_p_start = 1.5
        self.li_bic_p_stop = 2.5
        self.li_bic_p_num = 3

        #bic hac threshold for merging clusters
        self.bic_hac_start = 2.7
        self.bic_hac_end = 2.8
        self.bic_hac_num = 4

        #hac_iv threshold
        self.t_min = -10
        self.t_max = 20
        self.t_num = 6

        #viterbi penalty    
        self.vit_penalty = -250 
        
        self.wdir = os.path.join(self.dir, self.show)
        self.sdir = os.path.join(self.wdir, 'segments')
        self.pdir = os.path.join(self.wdir, 'plda')
        self.final_segments_dir = os.path.join(self.wdir, 'results')
        self.linear_bic_dir = os.path.join(self.wdir, 'linear.bic')
        self.bic_hac_dir = os.path.join(self.wdir, 'bic.hac')
        self.hac_iv_dir = os.path.join(self.wdir, 'hac.iv')
        self.results_vit_dir = os.path.join(self.wdir, 'results.vit')
        self.readme_dir = os.path.join(self.wdir, 'readme')
        self.mfcc_dir = os.path.join(self.wdir, 'mfcc')
        self.segments_dir = os.path.join(self.wdir, 'segments')
        self.plda_dir = os.path.join(self.wdir, 'plda')

        if not self.plda_dir:
            os.makedirs(self.plda_dir)
        if not self.segments_dir:
            os.makedirs(self.segments_dir)
        if not os.path.exists(self.readme_dir):
            os.makedirs(self.readme_dir)
        if not os.path.exists(self.wdir):
            os.makedirs(self.wdir)
        if not os.path.exists(self.sdir):
            os.makedirs(self.sdir)    
        if not os.path.exists(self.pdir):
            os.makedirs(self.pdir)
        if not os.path.exists(self.final_segments_dir):
            os.makedirs(self.final_segments_dir)
        if not os.path.exists(self.linear_bic_dir):
            os.makedirs(self.linear_bic_dir)    
        if not os.path.exists(self.bic_hac_dir):
            os.makedirs(self.bic_hac_dir)
        if not os.path.exists(self.hac_iv_dir):
            os.makedirs(self.hac_iv_dir)
        if not os.path.exists(self.results_vit_dir):
            os.makedirs(self.results_vit_dir)
        if not os.path.exists(self.mfcc_dir):
            os.makedirs(self.mfcc_dir)

        #Writing config file
        with open(os.path.join(self.readme_dir,'readme'), 'w+') as configfile:
            self.config.write(configfile)
        
        #Important filenames 
        self.input_show = os.path.join('source.files', show, 'audio', show + '.Mix-Headset.wav')    
        self.idmap_fn = self.show + '.idmap.h5'
        self.model_fn = 'data/model/ester_model_1024_300_150.h5'
        self.score_fn = os.path.join(self.plda_dir, self.show + '.score.plda.h5') 
        self.mfcc_fn = os.path.join(self.mfcc_dir, self.show + '.mfcc.h5')

        #Segmentation file's directories
        self.init_seg = os.path.join(self.segments_dir, 'init.seg')
        self.gd_seg = os.path.join(self.segments_dir, 'gd_seg_250.seg')
        self.linear_bic_seg = os.path.join(self.linear_bic_dir, 'li_bic.{:.2f}.seg')
        self.mfcc_m_speaker = os.path.join(self.wdir, self.mfcc_dir, self.show + '.{:.2f}.{:.2f}.test.mfcc.h5')
        self.bic_hac_seg = os.path.join(self.bic_hac_dir, 'bic_hac.{:.2f}.{:.2f}.seg')
        self.hac_iv_seg = os.path.join(self.final_segments_dir, 'bic_hac.{:.2f}.{:.2f}.{:.2f}.seg')
        self.results_vit_seg = os.path.join(self.results_vit_dir, 'bic_hac.{:.2f}.{:.2f}.{:.2f}.vit.-250.seg')
        self.input_seg = 'sad/input_oracle_sad_' + show + '.seg'

    
    def initialize_parameters(self, dict_params):
        for k,v in dict_params.items():
            setattr(self, k, v)

        
    def train(self):
        try:
            init_diar = Diar.read_seg(self.input_seg)
            #init_diar = segmentation.self.init_seg(cep, show)
            init_diar.pack(50)
            Diar.write_seg(self.init_seg, init_diar)
            gd_diar = segmentation.segmentation(self.cep, init_diar, self.win_size)
            Diar.write_seg(self.gd_seg, gd_diar)
        except Exception as e:
            traceback.print_exec()
            print("initialziation fault")

        
        #performing experiment     
        self.create_seg_bic_linear(self.cep, gd_diar)
        self.create_seg_bic_hac(self.cep, self.linear_bic_dir)
        self.create_seg_iv_AHC(self.bic_hac_dir, self.input_show)
        self.create_seg_viterbi(self.cep, self.hac_iv_dir)

    
    def train_ivectors(self, model_iv, segment_dir, filename, segmentation, input_show):
        try:
            fn = os.path.join(segment_dir,filename+'mfcc.per.speaker.h5')
            #print(os.getcwd())
            fe = get_feature_extractor(self.input_show, type_feature_extractor='basic')
            idmap_bic = fe.save_multispeakers(segmentation.id_map(), \
                                                    output_feature_filename=fn, keep_all=False)
            fs = get_feature_server(fn, 'sid')
            ivectors = model_iv.train(fs, idmap_bic)
            #print(ivectors)
            return model_iv      
        except Exception as e:
            traceback.print_exc()

    
    def score_plda(self, model_iv):
        try:
            distance = model_iv.score_plda_slow()
            distance.write(self.score_fn)
            scores = Scores(scores_file_name=self.score_fn)
            return scores
        except Exception as e:
            traceback.print_exc()

    
    def create_seg_bic_linear(self, cep, diar):
        for t1 in np.linspace(self.li_bic_p_start, self.li_bic_p_stop, self.li_bic_p_num):
            bicl_diar = segmentation.bic_linear(cep, diar, t1, sr=False)
            Diar.write_seg(self.linear_bic_seg.format(t1), bicl_diar)

    
    def create_seg_bic_hac(self, cep, segment_dir):
        for file_name in os.listdir(segment_dir):
            try:
                diar = Diar.read_seg(os.path.join(segment_dir, file_name))
                for bic_value in np.linspace(self.bic_hac_start, self.bic_hac_end, self.bic_hac_num):
                    bic = hac_bic.HAC_BIC(cep, diar, bic_value, sr=False)
                    bic_hac_diar = bic.perform(to_the_end=True)
                    Diar.write_seg(os.path.join(self.bic_hac_dir, file_name+'.bic_value.{:.2f}'.format(bic_value))\
                                , bic_hac_diar)
            except Exception as e:
                traceback.print_exc()
                continue

    
    def create_seg_iv_AHC(self, segment_dir, input_show):
        model_iv = ModelIV(self.model_fn)
        #print(segment_dir)
        for file_name in os.listdir(segment_dir):
            try:
                segment_diar = Diar.read_seg(os.path.join(segment_dir, file_name))
                #print(segment_diar)
                model = self.train_ivectors(model_iv, self.mfcc_dir,file_name, segment_diar, self.input_show)
                scores = self.score_plda(model)
                for hac_value in np.linspace(self.t_min, self.t_max, self.t_num):
                    diar_iv, _, _ = hac_iv(segment_diar, scores, threshold=hac_value)
                    Diar.write_seg(os.path.join(self.hac_iv_dir, file_name+'.hac_value.{:.2f}'.format(hac_value))\
                                , diar_iv)
            except Exception as e:
                traceback.print_exc()
                print("There is an error over here")
                continue


    def create_features(self):
        fe = get_feature_extractor(self.input_show, type_feature_extractor='basic')
        fe.save(self.show, output_feature_filename=self.mfcc_fn)
        fs = get_feature_server(self.mfcc_fn, feature_server_type='basic')
        self.cep, self.vad_labels = fs.load(self.show)
    

    def create_seg_viterbi(self, cep, segment_dir):
        #viterbi resegmentation
        for file_name in os.listdir(segment_dir):
            diar = Diar.read_seg(os.path.join(segment_dir, file_name))
            vit_diar = viterbi.viterbi_decoding(cep, diar, self.vit_penalty)
            Diar.write_seg(os.path.join(self.results_vit_dir, file_name+'.viterbi.{:.2f}'.format(-250)), vit_diar)                        