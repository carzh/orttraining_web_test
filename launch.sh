#! /bin/bash

rm -rf dist
rm -rf node_modules

rm package-lock.json

npm cache clean --force
npm cache verify

npm install

npx webpack
npx light-server -s . -p 8080
