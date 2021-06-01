### BERTje for Dutch 
<ul>
  <li> BERTje is the pre-trained monolingual BERT model for the Dutch language from the University of Groningen <br> </li>
  <li> The BERTje model is pre-trained using MLM and sentence order prediction (SOP) tasks with whole word masking (WWM). <br> </li>
  <li> The BERTje model is trained using several Dutch corpora, including TwNC (a Dutch news corpus), SoNAR-500 (a multi-genre reference corpus), Dutch Wikipedia          text, web news, and books <br> </li>
  <li> The model has been pre-trained for about 1 million iterations <br> </li>
</ul>

### FlauBERT for French
<ul>
  <li> FlauBERT, which stands for French Language Understanding via BERT, is a pre-trained BERT model for the French language <br> </li>
  <li> The FlauBERT model performs better than the multilingual and cross-lingual models on many downstream French NLP tasks <br> </li>
  <li> FlauBERT is trained on a huge heterogeneous French corpus. The French corpus consists of 24 sub-corpora containing data from various sources, including              Wikipedia, books, internal crawling, WMT19 data, French text from OPUS (an open source parallel corpus), and Wikimedia. <br> </li>
  <li> FlauBERT consists of a vocabulary of 50,000 tokens <br> </li>
  <li> FlauBERT is trained only on the MLM task and, while masking tokens, we use dynamic masking instead of static masking <br> </li>
</ul>

### BETO for Spanish
<ul>
  <li> BETO is the pre-trained BERT model for the Spanish language from the Universidad de Chile. It is trained using the MLM task with WWM <br> </li>
  <li> The configuration of BETO is the same as the standard BERT-base model <br> </li>
  <li> The researchers of BETO provided two variants of the BETO model, called BETO-cased and BETO-uncased, for cased and uncased text, respectively <br> </li>
  <li> The pre-trained BETO is open sourced, so we can download it directly and use it for downstream tasks <br> </li>
  <li> Researchers have also shown that the performance of BETO is better than M-BERT for many downstream task <br> </li>
</ul>
