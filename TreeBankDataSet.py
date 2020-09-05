
from torch.utils.data import Dataset
import nltk
from nltk.corpus import treebank
from nltk.corpus import brown
from collections import defaultdict
import torch



class CustomDataset(Dataset):
    
    def __init__(self,dname='treebank'):
        super().__init__()
        
        
        data = None
        #selecting the datset
        if dname =='treebank':
            if len(treebank.words()) == 0:    
                nltk.download('treebank')
            data = treebank.tagged_sents(tagset='universal')
            
        elif dname == 'brown':
            if len(brown.words()) == 0:    
                nltk.download('brown')
            data = brown.tagged_sents(tagset='universal')
            
        
        self.data=data
        #print(data[0:1])
        vocab,tags =self._build_vocab()
        max_sent_len = max(map(len, data))
        self.max_sent_len = max_sent_len
        self.word_to_idx = defaultdict(lambda:0, {word:idx for idx,word in enumerate(vocab)})
        self.idx_to_word = {idx:word for word,idx in self.word_to_idx.items()}
        self.tag_to_idx = {tag:idx for idx,tag in enumerate(tags)}
        self.idx_to_tag = {idx:tag for tag,idx in self.tag_to_idx.items()}
        self.sen_list,self.tag_list = self._convert_to_num()
        
    
    def get_target_size(self):
        return len(self.tag_to_idx)
    
    def get_vocab_size(self):
        return len(self.word_to_idx)
   
           
    
    def _convert_to_num(self):
        data = self.data
        max_sent_len = self.max_sent_len 
        sent_list=[]
        taggs_list=[]
           
        for sen in data:
            num_row = [self.word_to_idx[word.lower()] for word,tag in sen]
            tag_row = [self.tag_to_idx[tag] for word,tag in sen]
            num_row = num_row +[0]*(max_sent_len-len(num_row))
            tag_row = tag_row +[0]*(max_sent_len-len(tag_row))
            num_row = torch.tensor(num_row)
            tag_row = torch.tensor(tag_row)
            sent_list.append(num_row)
            taggs_list.append(tag_row)
            
        return sent_list,taggs_list
    
    
        
    def _build_vocab(self):
        data=self.data
        vocabset=set()
        tagset = set()
        all_sents_tags = [[(word.lower(), tag) for word, tag in sentence] for sentence in data]
        for sent_tag in all_sents_tags:
            sent,tags = zip(*sent_tag)
            vocabset.update(sent)
            tagset.update(tags)
        return ['<UNK>','<EOS>']+list(vocabset),list(tagset)
            
     
    def __len__(self):
        return len(self.sen_list)
    
    def __getitem__(self,idx):
        return self.sen_list[idx],self.tag_list[idx]
    

cdataset = CustomDataset(dname='brown')  
