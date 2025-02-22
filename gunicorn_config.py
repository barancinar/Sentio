timeout = 120  # Timeout süresini artır
workers = 2    # Worker sayısı
threads = 2    # Thread sayısı
worker_class = 'gthread'  # Thread bazlı worker
max_requests = 1000      # Her worker'ın maksimum istek sayısı
max_requests_jitter = 50 # Yeniden başlatma zamanlaması için jitter