server {
        listen 80;
        listen [::]:80;
        server_name messenger.plan4.house;

        location ~ /.well-known {
               root /home/sunnyville/content/well;
        }
        location / {
                return 301 https://messenger.plan4.house;
        }
}

server {
        listen 443 ssl http2;
        listen [::]:443 ssl http2;

        server_name messenger.plan4.house;

        ssl_certificate      /etc/letsencrypt/live/plan4.house/fullchain.pem;
        ssl_certificate_key  /etc/letsencrypt/live/plan4.house/privkey.pem;
	ssl_trusted_certificate /etc/letsencrypt/live/plan4.house/fullchain.pem;

        location ~ /.well-known {
               root /home/sunnyville/content/well;
        }
        location / {
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;

		proxy_set_header X-Forwarded-Host $host;
		proxy_set_header X-Forwarded-Server $host;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

		proxy_http_version 1.1;

		proxy_pass http://localhost:8091;
		proxy_redirect off;
		proxy_buffering off;
        }
}
