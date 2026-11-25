#!/usr/bin/env Rscript
# Run from the repository root. Install SuperExactTest before running.
suppressPackageStartupMessages(library(SuperExactTest))
frames <- lapply(c('TF', 'RBP'), function(kind) {
  path <- file.path('data', 'niranjan', paste0(kind, '_master_brain_csv'), 'All events.csv')
  df <- read.csv(path, stringsAsFactors=FALSE, check.names=FALSE)
  df <- df[!is.na(df[[kind]]) & nzchar(df[[kind]]) & !is.na(df$event_id) & nzchar(df$event_id), ]
  lapply(split(df$event_id, df[[kind]]), unique)
})
names(frames) <- c('TF', 'RBP')
universe <- unique(unlist(frames, use.names=FALSE))
n <- length(universe)
results <- vector('list', length(frames$TF) * length(frames$RBP))
i <- 0L
for (tf in names(frames$TF)) for (rbp in names(frames$RBP)) {
  a <- frames$TF[[tf]]; b <- frames$RBP[[rbp]]
  result <- SuperExactTest::MSET(list(a, b), n=n, lower.tail=FALSE, log.p=TRUE)
  observed <- length(result$intersects)
  # MSET returns 1 (not log(1)) for a zero overlap even with log.p=TRUE.
  logp <- if (observed == 0L) 0 else result$p.value
  i <- i + 1L
  results[[i]] <- data.frame(TF=tf, RBP=rbp, TF_events=length(a), RBP_events=length(b),
    shared_events=observed, background_events=n, expected_overlap=length(a)*length(b)/n,
    fold_enrichment=result$FE, log_p=logp)
}
out <- do.call(rbind, results)
# Adjust in log space so tiny p-values remain distinguishable.
ord <- order(out$log_p); m <- nrow(out)
ranked <- out$log_p[ord] + log(m / seq_len(m))
logq <- numeric(m); logq[ord] <- pmin(0, rev(cummin(rev(ranked))))
out$p_value <- exp(out$log_p)
out$minus_log10_p <- -out$log_p / log(10)
out$BH_fdr <- exp(logq)
out$minus_log10_BH_fdr <- -logq / log(10)
out$exploratory_fdr_below_0_05 <- logq < log(0.05)
out$log_p <- NULL
write.csv(out, 'graphs/niranjan/tables/tf_rbp_count_exact_test_exploratory.csv', row.names=FALSE)
cat(sprintf('SuperExactTest %s: %d background events; %d pairs\n', packageVersion('SuperExactTest'), n, m))
