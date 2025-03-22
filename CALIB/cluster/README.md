## Guides

### Connect to AKS and Peek at clusters
1. Login via az cli tool `az login` and select your subscription
    a. Verify you have the correct permissions to connect to the ask cluster.
2. Install the kubectl tool locally [docs to install](https://kubernetes.io/docs/tasks/tools/)
2. To merge your aks credentials into your local config, run: `az aks get-credentials --resource-group <RESOURCE_GROUP_NAME> --name <AKS_CLUSTER_NAME>`
    a. e.g.: `az aks get-credentials --resource-group rg-rayaks-sandbox-wus2-00 --name aks-rayaks-sandbox-wus2-00`
3. Inspect the pods `kubectl get pods`
    a. Inspect the nodes: `kubectl get nodes`
4. Forward the necessary port for mysql (this will allow you to use any mysql client to connect to the mysql instance running in the cluster):  
    `kubectl port-forward mysql-0 3306:3306 &`

### Submit a job using kubectl
0. Make sure your enviornment has optuna, pymysql install to run following  
    `pip3 install optuna, pymysql`
1. Setup environment vars: 
   ```bash
	export MYSQL_DB=optunaDatabase
	export MYSQL_ROOT_PASSWORD=<some password>
	export MYSQL_PASSWORD=<some super secret password>
	export MYSQL_USER=optuna
	export STUDY_NAME="<some study name>"
   ```
2. Create Study:   
   `envsubst < laser-study-creator-deploy-manifests.yaml | kubectl apply -f -`

3. Create workers:   
   `envsubst < laser-worker-deploy-manifests.yaml | kubectl apply -f -`

4. Delete Study / workers in case they are hanging around after completion:   
   `kubectl delete -f laser-study-creator-deploy-manifests.yaml`  
   `kubectl delete -f laser-worker-deploy-manifests.yaml`

5. Peek at studies through optuna cli:  
   `optuna studies --storage "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@localhost:3306/${MYSQL_DB}"`

5. Peek at studies for a trial through optuna cli:  
   `optuna trials --study $STUDY_NAME --storage "mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@localhost:3306/${MYSQL_DB}"`

6. Build / Push image (in case of any depending changes in images):  
   ```bash
    cd CALIB 
    docker build -t idm-docker-staging.packages.idmod.org/laser/laser-polio:latest -f Dockerfile ..
    docker push idm-docker-staging.packages.idmod.org/laser/laser-polio:latest
   ```