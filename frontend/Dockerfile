# Containerize the React frontend application

# Use the lightweight and secure Alpine image for Node.js
FROM node:24-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
# This is done separately to leverage Docker layer caching
COPY package*.json ./

# Install build tools AND the compatibility library needed for many native Node modules,
# install all npm dependencies, and then remove the build tools.
# We keep libc6-compat as it might be needed at runtime.
# RUN apk add --no-cache libc6-compat \
#     && apk add --no-cache --virtual .build-deps python3 make g++ \
#     && npm install \
#     && apk del .build-deps

# Copy the rest of the frontend app's source code
COPY . .

# Make port 5173 available to the world outside this container
EXPOSE 5173

# The command to run the Vite dev server
# We add "-- --host" to make it accessible outside the container.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
# CMD npm run dev -- --host 0.0.0.0