library(ggplot2)
library(reshape2)
library(dplyr)


setwd("/home/george/Desktop/extreme-gnns")
x = read.csv("results.csv")
x$dataset = as.character(x$dataset)

x[grepl("Amazon",x$dataset),"dataset"] = "Amazon"

levels(x$model) = c("GAT","GCN","SAGE") 

df = melt(x,c("model","dataset"))
levels(df$variable) = c("Extreme Nodes","Regular Nodes")
df$value = df$value*100


for(mod in levels(x$model)){
  df %>% filter(model==mod ) %>%
    ggplot(aes(y = value, x = variable,fill = variable))+
    geom_bar(stat = "identity", position = "dodge", width=0.4)+
    facet_wrap(~dataset, scales ="free")+ggtitle(mod)+
    theme(legend.position="none")+xlab("")+ylab("Gain in AUC %")
  ggsave(paste0("figures/res_",mod,".pdf"))  
}




