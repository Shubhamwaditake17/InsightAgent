# Deploy on AWS EC2 (Ubuntu)
1. Launch a t3.small, open inbound port 8000 (restrict to your IP) and 22.
2. `sudo apt update && sudo apt install -y docker.io docker-compose-v2 git && sudo usermod -aG docker $USER` (re-login).
3. `git clone https://github.com/Shubhamwaditake17/InsightAgent && cd InsightAgent`
4. `cp .env.example .env` and set GROQ_API_KEY.
5. `docker compose up -d --build`
6. `curl http://<ec2-ip>:8000/health` then POST to `/ask`.
