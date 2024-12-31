# vector-similarity-comparison
To check similarity results using high dimensional vectors(openai ada model) against low dimension vectors(sentence-transformers) with KNN and ANN using redis docker image and deploy code as image for further applications

###### I have used redis/redis-stack docker image as vector service.
###### I'm trying to read text data from CSV file and create embeddings using OpenAI's text-embedding-ada model(dimension - 1536) and Sentence-transformers all-MiniLM-L6-v2(dimension - 384).
###### Storing the created embeddings in redis and doing similairty check of given text against processed text with configurable parameters like,
- Algorithm - [KNN/ANN]
- Dimension - [Highers/Lower]
- k - no of scores to fetch

###### Exposed functionailty through FastAPI

##### Steps/Activites done:
1. docker network create redis-net #to create common network for services to use
2. docker run -d -p 6001:6379 -p 6002:8001 --net redis-net --name redis-app -e REDIS_ARGS="--requirepass mypassword" redis/redis-stack #to pull and run redis/redis-stack image under above created network and configure with custom ports with password enabled for redis instance
3. docker build -t my_app:1.1 . #to build custom image after adding required code(from directory with above code)
4. docker run -p 5005:5005 --net redis-net my_app:1.0 #to run built custom image under above netork using redis service
5. docker-compose -f vector-similarity-compose.yml up #to consume all images as a single application flow - to validate complete functionaility before deploying
6. docker login
7. docker tag my_app:1.0 sandeep721/my_app:1.0
8. docker push sandeep721/my_app:1.0 #to deploy image to public docker hub https://hub.docker.com/r/sandeep721/my_app
9. docker pull sandeep721/my_app #action to pull the created image

##### Sample output:
###### /add_data - to create embeddings and store them in redis creating right schema and index

###### /search - to give input text and addtional parameters to check similarity scores
1. below is example of similarity score with openai embeddings
![image](https://github.com/user-attachments/assets/80b0ab0f-9be8-4589-9939-7609c409a37f)![image](https://github.com/user-attachments/assets/e69ff925-74c9-41d0-b8f9-905ee5307014)
2. below is example of similarity score with sentence-transformer embeddings
![image](https://github.com/user-attachments/assets/cb8986bd-ede7-4412-ba31-9855d2c3500c)![image](https://github.com/user-attachments/assets/49606076-8236-4880-8853-1aca4b8e8130)



