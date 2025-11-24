@echo off
echo ========================================
echo Starting Stanford CoreNLP Server
echo ========================================
echo.
echo Server will start on port 9000
echo Keep this window open while using the dependency parser
echo Press Ctrl+C to stop the server
echo.
cd stanford-corenlp-4.5.10
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
pause
