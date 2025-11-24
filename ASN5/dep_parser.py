"""
Dependency Parsing with Stanford CoreNLP
=========================================

This module handles dependency parsing using Stanford CoreNLP
through NLTK's interface. Pretty straightforward once you get
the server running.

Requirements:
- Stanford CoreNLP 4.5.x (should be in stanford-corenlp-4.5.10 folder)
- Java 1.8+ installed on your machine
- NLTK with CoreNLP interface

Usage:
1. Start CoreNLP server (see instructions below)
2. Run this script: python dep_parser.py
"""

from nltk.parse.corenlp import CoreNLPDependencyParser


def getDependencyParse(sentence):
    """
    Get a dependency parse for any sentence using Stanford CoreNLP.

    Takes a sentence string and returns CoNLL-formatted output.
    The output has these columns: word, POS tag, head index, dep relation.

    Args:
        sentence: The sentence you want to parse (just a regular string)

    Returns:
        A string in CoNLL format, or an error message if something went wrong

    Example:
        >>> result = getDependencyParse("The cat sat on the mat.")
        >>> print(result)
        The     DT      2       det
        cat     NN      3       nsubj
        sat     VBD     0       ROOT
        ...
    """
    # first things first - try to connect to the server
    try:
        depParser = CoreNLPDependencyParser(url='http://localhost:9000')
    except Exception as connectionError:
        # server isn't running - give helpful instructions
        errorMessage = f"""
Error connecting to CoreNLP server: {connectionError}

Please ensure the CoreNLP server is running:
1. Open a terminal/command prompt
2. Navigate to the CoreNLP directory:
   cd ASN5/stanford-corenlp-4.5.10
3. Start the server:
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
4. Wait for "StanfordCoreNLPServer listening at..." message
5. Run this script again
"""
        return errorMessage

    # server is up, let's try to parse
    try:
        parseResult = depParser.raw_parse(sentence)
        parsedSentence = next(parseResult)

        # convert to 4-column CoNLL format
        # that's: word, pos, head, relation
        conllOutput = parsedSentence.to_conll(4)
        return conllOutput

    except StopIteration:
        # parser didn't return anything
        return "Error: No parse found for the sentence."
    except Exception as parseError:
        return f"Error parsing sentence: {parseError}"


def parseAndDisplay(sentence):
    """
    Parse a sentence and show results in a nice table format.

    This is just a prettier wrapper around getDependencyParse.
    """
    print(f"\nSentence: \"{sentence}\"")
    print("-" * 60)

    # set up our table header
    print(f"{'Word':<15} {'POS':<8} {'Head':<8} {'Relation':<15}")
    print("-" * 60)

    parseResult = getDependencyParse(sentence)

    # check if we got an error instead of actual results
    if parseResult.startswith("Error"):
        print(parseResult)
        return

    # parse the CoNLL output and display nicely
    for line in parseResult.strip().split('\n'):
        if line.strip():
            parts = line.split('\t')
            if len(parts) >= 4:
                word, pos, head, relation = parts[0], parts[1], parts[2], parts[3]
                print(f"{word:<15} {pos:<8} {head:<8} {relation:<15}")
            else:
                # weird format, just print it raw
                print(line)


def main():
    """Run the dependency parser demo."""
    print("=" * 60)
    print("Dependency Parsing with Stanford CoreNLP")
    print("=" * 60)

    # give clear setup instructions since this is the tricky part
    print("\n*** CoreNLP Server Setup Instructions ***")
    print("""
Before running this script, you need to start the CoreNLP server:

1. Open a NEW terminal/command prompt
2. Navigate to: ASN5/stanford-corenlp-4.5.10
3. Run: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
4. Wait for "StanfordCoreNLPServer listening at..." message
5. Keep that terminal open and run this script in another terminal
""")

    # our test sentences - including some tricky ambiguous ones
    testSentences = [
        "The cat sat on the mat.",
        "Flying planes can be dangerous.",
        "Amid the chaos I saw her duck.",
        "I made her duck orange sauce."
    ]

    print("\n" + "=" * 60)
    print("Parsing Test Sentences")
    print("=" * 60)

    # run each sentence through the parser
    for sentence in testSentences:
        parseAndDisplay(sentence)
        print()

    # show the raw format too for reference
    print("\n" + "=" * 60)
    print("Raw CoNLL Format Example")
    print("=" * 60)
    print("\nSentence: \"The cat sat on the mat.\"")
    print("\nCoNLL Format (word\\tPOS\\thead\\trelation):")
    print(getDependencyParse("The cat sat on the mat."))


if __name__ == "__main__":
    main()
