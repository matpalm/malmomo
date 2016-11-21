library(ggplot2)
library(plyr)
library(lubridate)
library(reshape)

# ./p/emit_stats.py runs/XX/out > /tmp/mt.tsv
df = read.delim("/tmp/mt.tsv", h=F, 
                col.names=c("run", "episode", "move_mu", "move_std", "turn_mu", "turn_std"))
df = melt(df, id=c("run", "episode"))
ggplot(df, aes(episode, value)) + 
  geom_point(alpha=0.5, aes(colour=run)) + geom_smooth(aes(colour=run)) +  
  facet_grid(~variable) + ylim(-1, 1)

# ./p/emit_rewards.py runs/XX/out > /tmp/r.tsv
df = read.delim("/tmp/r.tsv", h=F,  col.names=c("run", "episode", "rewards"))
ggplot(df, aes(episode, rewards)) + 
  geom_point(alpha=1.0, aes(colour=run)) + 
  geom_smooth(aes(colour=run))
