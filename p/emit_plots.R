library(ggplot2)
library(grid)
library(gridExtra)
library(reshape)

# ./p/emit_stats.py runs/XX/out > /tmp/mt.tsv
df = read.delim("/tmp/mt.tsv", h=F,
                col.names=c("run", "episode", "move_mu", "move_std", "turn_mu", "turn_std"))
df = melt(df, id=c("run", "episode"))
p1 <- ggplot(df, aes(episode, value)) +
  geom_point(alpha=0.1, aes(colour=run)) + geom_smooth(aes(colour=run)) +
  facet_grid(~variable)

# ./p/emit_rewards.py runs/XX/out > /tmp/r.tsv
df = read.delim("/tmp/r.tsv", h=F, col.names=c("run", "episode", "rewards"))
p2a <- ggplot(df, aes(episode, rewards)) +
  geom_point(alpha=1.0, aes(colour=run)) +
  geom_smooth(aes(colour=run))
#p2b <- ggplot(df, aes(episode, rewards)) +
#  geom_point(alpha=1.0, aes(colour=run)) +
#  geom_smooth(aes(colour=run)) + facet_grid(~run)

df = read.delim("/tmp/l.tsv", h=F, col.names=c("run", "episode", "median_loss"))
p3 <- ggplot(df, aes(episode, median_loss)) +
  geom_point(alpha=0.2, aes(colour=run)) +
  geom_smooth(aes(colour=run))


png("/tmp/plots.png", width=1000, height=600)
grid.arrange(p1, p2a, p3)
dev.off()
