from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('zhihan1996/DNA_bert_6').save_pretrained('/hps/nobackup/jlees/fierro/DNABERT-6')
BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6').save_pretrained('/hps/nobackup/jlees/fierro/DNABERT-6')
