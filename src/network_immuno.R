library(igraph)
library(tnet)

mhc<-read.csv('immuno_data.csv')
pep_encode<-factor(mhc$sequence,levels = unique(mhc$sequence),labels = 1:length(unique(mhc$sequence)))
hla_encode<-factor(mhc$mhc,levels = unique(mhc$mhc),labels = 1:length(unique(mhc$mhc)))
mhc$hla_encode<-hla_encode
mhc$pep_encode<-pep_encode
mhc<-mhc[c('mhc',"pep_encode",'hla_encode',"sequence",'Label')]
mhc$Label[which(mhc$Label==0)] = 10^(-10)
g<-graph.data.frame(mhc,directed = F) %>%set_edge_attr('weight',value = as.numeric( mhc$Label))
V(g)$type <- bipartite_mapping(g)$type
all_degree<-degree(g,normalized = T)
print('degree')
all_close<-closeness(g)
print('close')
all_between<-betweenness(g)
print('between')
all_evcent<-evcent(g,scale = F)$vector
print('evcent')
all_feature <- cbind(all_degree,all_close,all_between,all_evcent)
all_feature<-as.data.frame(all_feature);
hla_feature<-all_feature[grep('HLA',rownames(all_feature)),];hla_feature$mhc<-rownames(hla_feature);
colnames(hla_feature)<-gsub('all','hla',colnames(hla_feature))
pep_feature<-all_feature[-grep('HLA',rownames(all_feature)),];pep_feature$pep_encode<-rownames(pep_feature)
colnames(pep_feature)<-gsub('all','pep',colnames(pep_feature))
tmp <- merge(mhc,hla_feature,by = 'mhc')
tmp <- merge(tmp,pep_feature,by = 'pep_encode')
for (i in 6:13){
  tmp[,i] = scale(tmp[,i])
}
write.csv(tmp,file = 'immuno_train.csv')