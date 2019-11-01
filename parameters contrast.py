#coding-utf-8
before_quan =["conv.seq_module.0.weight", "conv.seq_module.0.bias", "conv.seq_module.1.weight", "conv.seq_module.1.bias", "conv.seq_module.1.running_mean", "conv.seq_module.1.running_var", "conv.seq_module.3.weight", "conv.seq_module.3.bias", "conv.seq_module.4.weight", "conv.seq_module.4.bias", "conv.seq_module.4.running_mean", "conv.seq_module.4.running_var", "rnns.0.rnn.weight_ih_l0", "rnns.0.rnn.weight_hh_l0", "rnns.0.rnn.bias_ih_l0", "rnns.0.rnn.bias_hh_l0", "rnns.0.rnn.weight_ih_l0_reverse", "rnns.0.rnn.weight_hh_l0_reverse", "rnns.0.rnn.bias_ih_l0_reverse", "rnns.0.rnn.bias_hh_l0_reverse", "rnns.1.rnn.weight_ih_l0", "rnns.1.rnn.weight_hh_l0", "rnns.1.rnn.bias_ih_l0", "rnns.1.rnn.bias_hh_l0", "rnns.1.rnn.weight_ih_l0_reverse", "rnns.1.rnn.weight_hh_l0_reverse", "rnns.1.rnn.bias_ih_l0_reverse", "rnns.1.rnn.bias_hh_l0_reverse", "rnns.2.rnn.weight_ih_l0", "rnns.2.rnn.weight_hh_l0", "rnns.2.rnn.bias_ih_l0", "rnns.2.rnn.bias_hh_l0", "rnns.2.rnn.weight_ih_l0_reverse", "rnns.2.rnn.weight_hh_l0_reverse", "rnns.2.rnn.bias_ih_l0_reverse", "rnns.2.rnn.bias_hh_l0_reverse", "rnns.3.rnn.weight_ih_l0", "rnns.3.rnn.weight_hh_l0", "rnns.3.rnn.bias_ih_l0", "rnns.3.rnn.bias_hh_l0", "rnns.3.rnn.weight_ih_l0_reverse", "rnns.3.rnn.weight_hh_l0_reverse", "rnns.3.rnn.bias_ih_l0_reverse", "rnns.3.rnn.bias_hh_l0_reverse", "rnns.4.rnn.weight_ih_l0", "rnns.4.rnn.weight_hh_l0", "rnns.4.rnn.bias_ih_l0", "rnns.4.rnn.bias_hh_l0", "rnns.4.rnn.weight_ih_l0_reverse", "rnns.4.rnn.weight_hh_l0_reverse", "rnns.4.rnn.bias_ih_l0_reverse", "rnns.4.rnn.bias_hh_l0_reverse", "fc.0.module.0.weight", "fc.0.module.0.bias"]

after_quan = ["module2.weight", "module2.bias", "module4.weight", "module4.bias", "module13.weight_ih_l0", "module13.weight_hh_l0", "module13.bias_ih_l0", "module13.bias_hh_l0", "module13.weight_ih_l0_reverse", "module13.weight_hh_l0_reverse", "module13.bias_ih_l0_reverse", "module13.bias_hh_l0_reverse", "module21.weight_ih_l0", "module21.weight_hh_l0", "module21.bias_ih_l0", "module21.bias_hh_l0", "module21.weight_ih_l0_reverse", "module21.weight_hh_l0_reverse", "module21.bias_ih_l0_reverse", "module21.bias_hh_l0_reverse", "module29.weight_ih_l0", "module29.weight_hh_l0", "module29.bias_ih_l0", "module29.bias_hh_l0", "module29.weight_ih_l0_reverse", "module29.weight_hh_l0_reverse", "module29.bias_ih_l0_reverse", "module29.bias_hh_l0_reverse", "module37.weight_ih_l0", "module37.weight_hh_l0", "module37.bias_ih_l0", "module37.bias_hh_l0", "module37.weight_ih_l0_reverse", "module37.weight_hh_l0_reverse", "module37.bias_ih_l0_reverse", "module37.bias_hh_l0_reverse", "module45.weight_ih_l0", "module45.weight_hh_l0", "module45.bias_ih_l0", "module45.bias_hh_l0", "module45.weight_ih_l0_reverse", "module45.weight_hh_l0_reverse", "module45.bias_ih_l0_reverse", "module45.bias_hh_l0_reverse", "module57.weight", "module57.bias"]
print(len(before_quan)) #54
print(len(after_quan))  #46
'''
[ "module13.weight_ih_l0", "module13.weight_hh_l0", "module13.bias_ih_l0", "module13.bias_hh_l0",
  "module13.weight_ih_l0_reverse", "module13.weight_hh_l0_reverse", "module13.bias_ih_l0_reverse", "module13.bias_hh_l0_reverse"]

["rnns.0.rnn.weight_ih_l0", "rnns.0.rnn.weight_hh_l0", "rnns.0.rnn.bias_ih_l0", "rnns.0.rnn.bias_hh_l0",
 "rnns.0.rnn.weight_ih_l0_reverse", "rnns.0.rnn.weight_hh_l0_reverse", "rnns.0.rnn.bias_ih_l0_reverse", "rnns.0.rnn.bias_hh_l0_reverse"]
数据的顺序是ifgo
'''

quantconfig={   "rnns.0.rnn.weight_ih_l0":[16,15],
    "rnns.0.rnn.weight_hh_l0":[16,14],
    "rnns.0.rnn.bias_ih_l0":[16,6],
    "rnns.0.rnn.weight_ih_l0_reverse":[16,15],
    "rnns.0.rnn.weight_hh_l0_reverse":[16,15],
    "rnns.0.rnn.bias_ih_l0_reverse":[16,6],
    "rnns.1.rnn.weight_ih_l0":[16,15],
    "rnns.1.rnn.weight_hh_l0":[16,15],
    "rnns.1.rnn.bias_ih_l0":[16,8],
    "rnns.1.rnn.weight_ih_l0_reverse":[16,15],
    "rnns.1.rnn.weight_hh_l0_reverse":[16,15],
    "rnns.1.rnn.bias_ih_l0_reverse":[16,8],
    "rnns.2.rnn.weight_ih_l0":[16,15],
    "rnns.2.rnn.weight_hh_l0":[16,14],
    "rnns.2.rnn.bias_ih_l0":[16,8],
    "rnns.2.rnn.weight_ih_l0_reverse":[16,16],
    "rnns.2.rnn.weight_hh_l0_reverse":[16,15],
    "rnns.2.rnn.bias_ih_l0_reverse":[16,8],
    "rnns.3.rnn.weight_ih_l0":[16,16],
    "rnns.3.rnn.weight_hh_l0":[16,15],
    "rnns.3.rnn.bias_ih_l0":[16,8],
    "rnns.3.rnn.weight_ih_l0_reverse":[16,16],
    "rnns.3.rnn.weight_hh_l0_reverse":[16,15],
    "rnns.3.rnn.bias_ih_l0_reverse":[16,8],
    "rnns.4.rnn.weight_ih_l0":[16,14],
    "rnns.4.rnn.weight_hh_l0":[16,14],
    "rnns.4.rnn.bias_ih_l0":[16,8],
    "rnns.4.rnn.weight_ih_l0_reverse":[16,15],
    "rnns.4.rnn.weight_hh_l0_reverse":[16,14],
    "rnns.4.rnn.bias_ih_l0_reverse":[16,8],
}
weight_ih_l0=[15,15,15,16,14]
weight_hh_l0=[14,15,14,15,14]
weight_ih_l0_reverse=[15,15,16,16,15]
weight_hh_l0_reverse=[15,15,15,15,14]
bias_ih_l0=[6,8,8,8,8]
bias_ih_l0_reverse=[6,8,8,8,8]

# 13 21 29 37 45   一共五层lstm 每层
#def output_file(name,param):
'''
ih = []
hh =[]
bias_ih=[]
bias_hh =[]
ih_inv = []
hh_inv =[]
bias_ih_inv=[]
bias_hh_inv =[]
s=['ih',
'hh',
'bias_ih',
'bias_hh',
'ih_inv',
'hh_inv',
'bias_ih_inv',
'bias_hh_inv']
ls =[ ih ,
 hh ,
 bias_ih ,
 bias_hh ,
 ih_inv ,
 hh_inv ,
 bias_ih_inv ,
 bias_hh_inv ]
for i in range (5):
    ih.append(8*i+4)
    hh.append(8*i+5)
    bias_ih.append(8*i+6)
    bias_hh.append(8*i+7)
    ih_inv.append(8*i+8)
    hh_inv.append(8*i+9)
    bias_ih_inv.append(8*i+10)
    bias_hh_inv.append(8*i+11)
    
for i in range(len(s)):
    print(s[i]+'=')
    print(ls[i])
'''

s=['ih',
'hh',
'bias_ih',
'bias_hh',
'ih_inv',
'hh_inv',
'bias_ih_inv',
'bias_hh_inv']

ih=[4, 12, 20, 28, 36]
hh=[5, 13, 21, 29, 37]
bias_ih=[6, 14, 22, 30, 38]
bias_hh=[7, 15, 23, 31, 39]
ih_inv=[8, 16, 24, 32, 40]
hh_inv=[9, 17, 25, 33, 41]
bias_ih_inv=[10, 18, 26, 34, 42]
bias_hh_inv=[11, 19, 27, 35, 43]

lstm =[13,21, 29,37,45]
for i in ih:
    print(after_quan[i])

paramlist =[]
for name, param in model.named_parameters():
   print(name, param.shape)
   paramlist.append(param)

import numpy as np

def __amplify_data(data,bn,fp,method=1):
    #1 for floor, 2 for dpu round; use number, not amplified
    if method==1:
        data=np.floor(data*(2**fp))
        data=np.minimum(data,(2**(bn-1)-1))
        data=np.maximum(data,-(2**(bn-1)))
    elif method==2:
        data=data*(2**fp)
        data=np.minimum(data,(2**(bn-1)-1))
        data=np.maximum(data,-(2**(bn-1)))
        data=np.where(np.logical_and(data<0,(data-np.floor(data))==0.5),np.ceil(data),np.round(data))
    return data

def quantize_data(data,bn,fp,method=1):
    data=__amplify_data(data,bn,fp,method)
    return data/(2**fp)

def quantize_data2int16(data,bn,fp,method=1):
    data=__amplify_data(data,bn,fp,method)
    return data.astype(np.int16)



#np.array(i)
def output(paramlist,para_position,name,biasweight,forward,quant):
    for j in range(len(para_position)):
        parameter = paramlist[para_position[j]]
        i=parameter[:800,:]
        i=i.reshape(-1)
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'/weight_'+name+'_i'+'_layer'+str(j)+'.txt', quantize_data2int16(i.cpu().numpy(),16,quant[j],method=1))
        f=parameter[800:1600,:]
        f = f.reshape(-1)
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'/weight_'+name+'_f'+'_layer'+str(j)+'.txt', quantize_data2int16(f.cpu().numpy(),16,quant[j],method=1))
        g=parameter[1600:2400,:]
        g = g.reshape(-1)
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'/weight_'+name+'_g'+'_layer'+str(j)+'.txt', quantize_data2int16(g.cpu().numpy(),16,quant[j],method=1))
        o=parameter[2400:3200,:]
        o = o.reshape(-1)
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'/weight_'+name+'_o'+'_layer'+str(j)+'.txt', quantize_data2int16(o.cpu().numpy(),16,quant[j],method=1))

output(paramlist,ih,'ih','weight','forward',weight_ih_l0)
output(paramlist,hh,'hh','weight','forward',weight_hh_l0)
output(paramlist,ih_inv,'ih_inv','weight','backward',weight_ih_l0_reverse)
output(paramlist,hh_inv,'hh_inv','weight','backward',weight_hh_l0_reverse)

def bias(biasih,biashh,paramlist,biasweight,forward,quant):
    for j in range(len(biasih)):
        parameter = paramlist[biasih[j]]+paramlist[biashh[j]]
        i=parameter[:800]
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'bias_i'+'_layer'+str(j)+'.txt', quantize_data2int16(i.cpu().numyp(),16,quant[j],method=1))
        f=parameter[800:1600]
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'bias_f'+'_layer'+str(j)+'.txt', quantize_data2int16(f.cpu().numyp(),16,quant[j],method=1))
        g=parameter[1600:2400]
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'bias_g'+'_layer'+str(j)+'.txt', quantize_data2int16(g.cpu().numyp(),16,quant[j],method=1))
        o=parameter[2400:3200]
        np.savetxt('model_data_to_hardware/'+biasweight+'/'+forward+'bias_o'+'_layer'+str(j)+'.txt', quantize_data2int16(o.cpu().numyp(),16,quant[j],method=1))

bias(bias_ih,bias_hh,paramlist,'bias','forward',bias_ih_l0)
bias(bias_ih_inv,bias_hh_inv,paramlist,'backward',bias_ih_l0_reverse)


        from opts import add_decoder_args, add_inference_args
        from data.data_loader import SpectrogramParser
        import argparse

        parser = argparse.ArgumentParser(description='DeepSpeech transcription')
        parser = add_inference_args(parser)
        parser.add_argument('--audio-path', default='audio.wav',
                            help='Audio file to predict on')
        parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
        parser = add_decoder_args(parser)
        args = parser.parse_args()


        parser = SpectrogramParser(model.audio_conf, normalize=True)