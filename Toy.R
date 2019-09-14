library(splatter)
library(scater)

params <- newSplatParams()
params <- setParams(params, update = list(nGenes = 10000))

params <- setParams(params, update = list(batchCells = c(4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000)))
params <- setParams(params, update = list(batch.facLoc = c(0.5)))
params <- setParams(params, update = list(batch.facScale = c(0.5)))
                    
params <- setParams(params, update = list(group.prob = c(0.2, 0.2, 0.2, 0.2, 0.2)))

sim <- splatSimulate(params, verbose = FALSE, method = "groups")
sim <- normalize(sim)

plotPCA(sim, colour_by = "Group")

write.table(x = counts(sim), file = "./data/toy/toy.csv", sep = ",")
write.table(x = sim@colData@listData[["Batch"]], file = "./data/toy/toy_batch.csv", sep = ',')
write.table(x = sim@colData@listData[["Group"]], file = "./data/toy/toy_celltype.csv", sep = ',')
