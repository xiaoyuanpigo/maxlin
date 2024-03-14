import os
os.system("cp -f pool_maxlin.py alpha-beta-CROWN/complete_verifier/auto_LiRPA/operators/pooling.py")

#results
for i in [10,11,12,13,14]:
    eps=i*0.01
    command = f"python abcrown.py --config verivital{i}.yaml  2>&1  |tee figure5_{eps}.log"
    os.system(command)