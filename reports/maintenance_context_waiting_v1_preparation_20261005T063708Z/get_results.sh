mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial -e "ssh -p 2222" \
  pod-4ae4e7c0-13f9-4e53-8722-c363a3558c84@ssh.gputw.ai:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
