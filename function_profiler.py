import pstats
stats = pstats.Stats("profile_speed")
stats.sort_stats("tottime")

stats.print_stats(50)