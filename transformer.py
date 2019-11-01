

# A*B (dimenstion of m*n and n*k) on count m*n*k
#softmax one line count 1
#norm one time count 1
class transformer_flops():
    def __init__(self,dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab):
        self.dk=dk
        self.dv=dv
        self.seq=seq
        self.dmodel=dmodel
        self.h=h
        self.N=N
        self.d_ff=d_ff
        self.outlen = outlength
        self.vocab=vocab
        # h head number
        #dk dv  Q K V dim
        #seq sequence length
        # N  number of multihead-attention
        #d_ff dimension of the middle layer of feedforward layer
        # dmodel  dimension of word embedding
        #outlen output translation sequence length
        

    def attention(self,n,num_word,decoder):
        #X*Wq Wk Wv->Q K V
        Cal_Q = n*self.dmodel*self.dk
        Cal_K = num_word*self.dmodel*self.dk
        Cal_V = num_word*self.dmodel*self.dv

        #FIX ME
        # softmax(Q*KT/dk**1/2)*V  simplified as Q*KT*V  ignore softmax division and transposition
        Cal_QKt = n*self.dk*num_word
        total_soft = n
        division = n*num_word

        #just multiply
        mask = 0
        if decoder:
            mask=n*num_word

        Cal_qktv = n*num_word*self.dv

        total_mut = Cal_Q+Cal_K+Cal_V+Cal_QKt+Cal_qktv+mask +division
        total_add = Cal_Q+Cal_K+Cal_V+Cal_QKt+Cal_qktv

        return total_mut,total_add,total_soft




    def mutihead(self,n,num_word,decoder):


        (total_mut,total_add,total_soft) = self.attention(n,num_word,decoder)

        #concat*W0
        op = n*self.dv*self.h*self.dmodel

        total_add = self.h*total_add + op
        total_mut = self.h*total_mut + op

        return total_mut,total_add,self.h*total_soft

    def encoder(self):

        #FIX mek
        # ignore pos embedding + embedding        it's small

        #mutihead attention
        (total_mut,total_add,total_soft) = self.mutihead(self.seq, self.seq, False)
        #print('encoder one multihead(total_mut:%s,total_add:%s,total_soft:%s)' % (total_mut, total_add, total_soft))
        #2-layer FC

        Cal_fc1= self.seq*self.dmodel*self.d_ff
        Cal_fc2= self.seq*self.d_ff*self.dmodel
        
        #RELU 
        total_relu = self.seq*self.d_ff*self.N
        #2 times add&norm
        add_norm = 2 * self.seq * self.dmodel

        total_mut =(Cal_fc1+Cal_fc2+total_mut)*self.N
        total_add =(Cal_fc1+Cal_fc2+total_add+add_norm)*self.N
        total_soft = total_soft*self.N
        total_norm = add_norm*self.N

        print('ENCODER----->add op: %s|mut op %s|softmax %s| layer normalization %s| relu %s' % (total_add, total_mut, total_soft, total_norm, total_relu))
        return total_mut,total_add,total_soft,total_norm,total_relu

    def decoder(self):
        #ignore dropout
        
        #masked multihead
        total_mut, total_add, total_soft, total_norm, total_relu=0,0,0,0,0
        for n in range(1,self.outlen+1):
            #masked
            (mut, add, soft) = self.mutihead(n, n, True)
            # encoder for kv
            (mut1, add1, soft1) = self.mutihead(n, self.seq, False)
            norm_and_add = 3 * n * self.dmodel
            Cal_fc1 = n * self.dmodel * self.d_ff
            Cal_fc2 = n * self.d_ff * self.dmodel
            
            #output layer
            out = self.dmodel*self.vocab
            
            total_mut = total_mut + mut + mut1 + Cal_fc1 + Cal_fc2
            total_add = total_add + add + add1 + norm_and_add + Cal_fc1 + Cal_fc2
            total_soft = total_soft + soft + soft1 
            total_norm = total_norm + norm_and_add 
            total_relu = total_relu + n * self.d_ff

        total_mut, total_add, total_soft, total_norm, total_relu=total_mut * self.N + out, total_add * self.N + out, total_soft * self.N + 1, total_norm * self.N, total_relu * self.N
        print('DECODER----->add op: %s|mut op %s|softmax %s| layer normalization %s| relu %s' % (total_add, total_mut, total_soft, total_norm, total_relu))
        return total_mut, total_add, total_soft, total_norm, total_relu
    
    def summary(self):

        (mute, adde, softe, norme, relue) = self.encoder()
        (mut, add, soft, norm, relu) = self.decoder()
        total_mut = mute+mut
        total_add = adde + add
        total_soft = soft +softe
        total_norm = norm +norme
        total_relu = relu + relue
        print('Summary----->add op: %s|mut op %s|softmax %s| layer normalization %s| relu %s'%(total_add,total_mut,total_soft,total_norm,total_relu))

        return total_mut,total_add,total_soft,total_norm,total_relu


class bert_op(transformer_flops):
    def __init__(self,dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab):
        super().__init__(dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab)
    def bert(self):
        return self.encoder()

if __name__=='__main__':
    dk= 64
    dv= 64
    seq= 32
    dmodel= 512
    h= 8
    N= 6
    d_ff = 2048
    outlength = 32
    vocab_size = 37000
    trans=transformer_flops(dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab_size)
    a=trans.summary()
    #print(op.attention(seq,seq,False))

    # bert base
    N = 12
    h = 12
    dmodel = 768
    d_ff = 4*dmodel
    kv =kv = dmodel/h

    bert = bert_op(dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab_size)
    bert.bert()

    #bert large
    N = 24
    h = 16
    dmodel = 1024
    d_ff = 4 * dmodel
    bertlarge = bert_op(dk,dv,seq,dmodel,h,N,d_ff,outlength,vocab_size)
    bertlarge.bert()
        



