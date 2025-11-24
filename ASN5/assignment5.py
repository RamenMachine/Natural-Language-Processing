"""
Constituency and Dependency Parsing
====================================

This project covers parsing techniques in NLP:
- Constituency Tree Visualization
- CKY Parsing Algorithm Implementation
- Dependency Parsing with Stanford CoreNLP
- Ambiguous Sentence Analysis
"""

import nltk
from nltk import Tree
from nltk.grammar import CFG, Nonterminal
import warnings
warnings.filterwarnings('ignore')


def downloadNltkData():
    """Grab all the NLTK data packages we need for this to work."""
    # these are the packages we'll be using throughout
    requiredPackages = ['punkt', 'averaged_perceptron_tagger', 'large_grammars']

    # loop through and download anything that's missing
    for packageName in requiredPackages:
        try:
            # nltk stores different types of data in different folders
            if packageName == 'punkt':
                nltk.data.find(f'tokenizers/{packageName}')
            elif packageName == 'averaged_perceptron_tagger':
                nltk.data.find(f'taggers/{packageName}')
            else:
                nltk.data.find(f'grammars/{packageName}')
        except LookupError:
            # couldn't find it, so let's download it
            print(f"Downloading {packageName}...")
            nltk.download(packageName, quiet=True)


# ============================================================================
# TASK 1: Constituency Tree Visualization
# ============================================================================

def createConstituencyTree():
    """
    Build a constituency tree for "Cat sat on the mat" using production rules.

    Here's what we're working with:
    S -> VP
    VP -> NP V PP
    NP -> DET ADJ N | DET N | N
    PP -> P NP

    Returns the completed parse tree as an nltk.Tree object.
    """
    # alright, let's break down the sentence piece by piece
    # Cat = noun, sat = verb, on = preposition, the = determiner, mat = noun

    # we're building this tree from the bottom up
    # first the leaves, then the branches, then the trunk

    # "Cat" is just a noun by itself, so NP -> N
    catNounPhrase = Tree('NP', [Tree('N', ['Cat'])])

    # "the mat" uses the rule NP -> DET N
    matNounPhrase = Tree('NP', [
        Tree('DET', ['the']),
        Tree('N', ['mat'])
    ])

    # "on the mat" is a prepositional phrase: PP -> P NP
    prepPhrase = Tree('PP', [
        Tree('P', ['on']),
        matNounPhrase
    ])

    # now we put together the verb phrase: VP -> NP V PP
    verbPhrase = Tree('VP', [
        catNounPhrase,
        Tree('V', ['sat']),
        prepPhrase
    ])

    # finally, the whole sentence: S -> VP
    fullTree = Tree('S', [verbPhrase])

    return fullTree


def visualizeConstituencyTree(parseTree, saveToFile=True):
    """
    Display the constituency tree in a nice readable format.
    Shows both text and pretty-printed versions.
    """
    print("=" * 60)
    print("TASK 1: Constituency Tree for 'Cat sat on the mat'")
    print("=" * 60)

    # let's show which production rules we actually used
    print("\nProduction Rules Used:")
    print("  S -> VP")
    print("  VP -> NP V PP")
    print("  NP -> N (for 'Cat')")
    print("  PP -> P NP")
    print("  NP -> DET N (for 'the mat')")

    # show the raw tree structure first
    print("\nTree Structure (text format):")
    print(parseTree)

    # now the pretty version that's easier to read
    print("\nPretty Print:")
    parseTree.pretty_print()

    # give instructions for graphical display
    # we skip the actual draw() call since it blocks the script
    print("\nTo display graphical tree, run in Python shell:")
    print("  >>> from assignment5 import createConstituencyTree")
    print("  >>> tree = createConstituencyTree()")
    print("  >>> tree.draw()")
    print("\n(Skipping tree.draw() in script mode to avoid blocking)")

    return parseTree


# ============================================================================
# TASK 2: CKY Parsing Algorithm Implementation
# ============================================================================

class CkyParser:
    """
    CKY (Cocke-Kasami-Younger) Parsing Algorithm.

    This is based on the algorithm from Jurafsky & Martin, Section 13.4.
    It uses dynamic programming to efficiently parse sentences and
    keeps track of back-pointers so we can reconstruct the parse trees.
    """

    def __init__(self, inputGrammar):
        """Set up the parser with a grammar, converting it to CNF first."""
        # keep the original around just in case we need it
        self.originalGrammar = inputGrammar

        # convert to Chomsky Normal Form - CKY only works with CNF
        self.grammar = self._convertToChomskyNormalForm(inputGrammar)

        # build lookup tables for faster parsing
        self._buildGrammarIndices()

    def _convertToChomskyNormalForm(self, inputGrammar):
        """
        Convert the grammar to Chomsky Normal Form.

        In CNF, every rule is either:
        - A -> B C (two non-terminals)
        - A -> a (one terminal)

        This makes the CKY algorithm work properly.
        """
        from nltk import CFG
        from nltk.grammar import Production

        # grab all the existing productions
        allProductions = list(inputGrammar.productions())
        startSymbol = inputGrammar.start()
        newProductions = []

        # we'll need to create new non-terminals for binarization
        helperCounter = 0

        def createHelperNonterminal():
            """Make a new unique non-terminal symbol."""
            nonlocal helperCounter
            helperCounter += 1
            return Nonterminal(f'_X{helperCounter}')

        # process each production rule
        for prod in allProductions:
            leftSide = prod.lhs()
            rightSide = prod.rhs()

            # skip empty productions - they mess things up
            if len(rightSide) == 0:
                continue

            # single symbol on right side - already good
            elif len(rightSide) == 1:
                newProductions.append(prod)

            # two symbols - might need to wrap terminals
            elif len(rightSide) == 2:
                processedRight = []
                for symbol in rightSide:
                    if isinstance(symbol, str):
                        # terminals in binary rules need their own non-terminal
                        helperNt = createHelperNonterminal()
                        newProductions.append(Production(helperNt, (symbol,)))
                        processedRight.append(helperNt)
                    else:
                        processedRight.append(symbol)
                newProductions.append(Production(leftSide, tuple(processedRight)))

            # more than two symbols - need to binarize
            else:
                # first, handle any terminals
                processedRight = []
                for symbol in rightSide:
                    if isinstance(symbol, str):
                        helperNt = createHelperNonterminal()
                        newProductions.append(Production(helperNt, (symbol,)))
                        processedRight.append(helperNt)
                    else:
                        processedRight.append(symbol)

                # now chain them together two at a time
                # A -> B C D E becomes A -> B X1, X1 -> C X2, X2 -> D E
                currentLeft = leftSide
                for idx in range(len(processedRight) - 2):
                    helperNt = createHelperNonterminal()
                    newProductions.append(Production(currentLeft, (processedRight[idx], helperNt)))
                    currentLeft = helperNt

                # don't forget the last pair
                newProductions.append(Production(currentLeft, (processedRight[-2], processedRight[-1])))

        return CFG(startSymbol, newProductions)

    def _buildGrammarIndices(self):
        """
        Create lookup tables for quick grammar access.

        We make two dictionaries:
        - terminalRules: maps terminals to their possible non-terminals
        - binaryRules: maps (B, C) pairs to possible A where A -> B C
        """
        self.terminalRules = {}
        self.binaryRules = {}

        # go through every production and index it
        for production in self.grammar.productions():
            rightSide = production.rhs()
            leftSide = production.lhs()

            # terminal rule: A -> 'word'
            if len(rightSide) == 1 and isinstance(rightSide[0], str):
                terminalWord = rightSide[0].lower()
                if terminalWord not in self.terminalRules:
                    self.terminalRules[terminalWord] = []
                self.terminalRules[terminalWord].append((leftSide, production))

            # binary rule: A -> B C
            elif len(rightSide) == 2 and all(isinstance(s, Nonterminal) for s in rightSide):
                pairKey = (rightSide[0], rightSide[1])
                if pairKey not in self.binaryRules:
                    self.binaryRules[pairKey] = []
                self.binaryRules[pairKey].append((leftSide, production))

    def parse(self, sentence):
        """
        Parse a sentence using the CKY algorithm.

        This is the main parsing function. It fills in a chart bottom-up
        and then extracts all valid parse trees from it.

        Returns a list of nltk.Tree objects (could be empty if no parse).
        """
        # handle both string input and word lists
        if isinstance(sentence, str):
            wordList = sentence.lower().split()
        else:
            wordList = [w.lower() for w in sentence]

        numWords = len(wordList)

        # empty sentence = nothing to do
        if numWords == 0:
            return []

        # set up the CKY chart
        # chart[i][j] maps non-terminals to their back-pointers
        chart = [[{} for _ in range(numWords + 1)] for _ in range(numWords + 1)]

        # fill in the diagonal first - these are single words
        for wordIdx in range(1, numWords + 1):
            currentWord = wordList[wordIdx - 1]

            # look up what non-terminals can produce this word
            if currentWord in self.terminalRules:
                for nonTerminal, prod in self.terminalRules[currentWord]:
                    if nonTerminal not in chart[wordIdx-1][wordIdx]:
                        chart[wordIdx-1][wordIdx][nonTerminal] = []
                    chart[wordIdx-1][wordIdx][nonTerminal].append(('terminal', currentWord))

            # handle any unary rules that apply
            self._processUnaryRules(chart[wordIdx-1][wordIdx])

        # now fill in the rest of the chart, bottom-up
        for spanLength in range(2, numWords + 1):
            for startPos in range(numWords - spanLength + 1):
                endPos = startPos + spanLength

                # try all possible split points
                for splitPoint in range(startPos + 1, endPos):
                    leftCell = chart[startPos][splitPoint]
                    rightCell = chart[splitPoint][endPos]

                    # check every combination of left and right non-terminals
                    for leftNt in leftCell:
                        for rightNt in rightCell:
                            pairKey = (leftNt, rightNt)
                            if pairKey in self.binaryRules:
                                for parentNt, prod in self.binaryRules[pairKey]:
                                    if parentNt not in chart[startPos][endPos]:
                                        chart[startPos][endPos][parentNt] = []
                                    chart[startPos][endPos][parentNt].append(
                                        ('binary', splitPoint, leftNt, rightNt)
                                    )

                # handle unary rules for this cell too
                self._processUnaryRules(chart[startPos][endPos])

        # check if we got a complete parse
        startSymbol = self.grammar.start()
        if startSymbol not in chart[0][numWords]:
            return []

        # extract the actual trees from the back-pointers
        resultTrees = self._reconstructTrees(chart, 0, numWords, startSymbol, wordList)
        return resultTrees

    def _processUnaryRules(self, chartCell):
        """
        Add any non-terminals reachable through unary rules.

        We keep going until nothing new is added (fixed point).
        """
        madeChanges = True
        while madeChanges:
            madeChanges = False
            newEntries = {}

            for existingNt in list(chartCell.keys()):
                # look for rules A -> B where B is what we have
                for prod in self.grammar.productions():
                    rightSide = prod.rhs()
                    leftSide = prod.lhs()

                    if len(rightSide) == 1 and rightSide[0] == existingNt:
                        if leftSide not in chartCell and leftSide not in newEntries:
                            newEntries[leftSide] = [('unary', existingNt)]
                            madeChanges = True

            chartCell.update(newEntries)

    def _reconstructTrees(self, chart, startIdx, endIdx, nonTerminal, wordList, maxTrees=10):
        """
        Build actual parse trees from the back-pointers in the chart.

        This is recursive - we follow the pointers all the way down.
        We limit to maxTrees to avoid explosion with ambiguous grammars.
        """
        if nonTerminal not in chart[startIdx][endIdx]:
            return []

        resultTrees = []

        for backPointer in chart[startIdx][endIdx][nonTerminal]:
            # stop if we've got enough trees
            if len(resultTrees) >= maxTrees:
                break

            pointerType = backPointer[0]

            if pointerType == 'terminal':
                # leaf node - just the word
                word = backPointer[1]
                newTree = Tree(str(nonTerminal), [word])
                resultTrees.append(newTree)

            elif pointerType == 'binary':
                # binary rule - recursively build both children
                splitAt, leftNt, rightNt = backPointer[1], backPointer[2], backPointer[3]

                leftSubtrees = self._reconstructTrees(chart, startIdx, splitAt, leftNt, wordList, maxTrees)
                rightSubtrees = self._reconstructTrees(chart, splitAt, endIdx, rightNt, wordList, maxTrees)

                # combine all possibilities
                for leftTree in leftSubtrees:
                    for rightTree in rightSubtrees:
                        if len(resultTrees) >= maxTrees:
                            break
                        newTree = Tree(str(nonTerminal), [leftTree, rightTree])
                        resultTrees.append(newTree)

            elif pointerType == 'unary':
                # unary rule - just one child
                childNt = backPointer[1]
                childTrees = self._reconstructTrees(chart, startIdx, endIdx, childNt, wordList, maxTrees)

                for childTree in childTrees:
                    if len(resultTrees) >= maxTrees:
                        break
                    newTree = Tree(str(nonTerminal), [childTree])
                    resultTrees.append(newTree)

        return resultTrees

    def parseAndPrint(self, sentence):
        """Parse a sentence and display the results nicely."""
        print(f"\nParsing: \"{sentence}\"")
        print("-" * 50)

        resultTrees = self.parse(sentence)

        if not resultTrees:
            print("No valid parse found.")
            return None

        # show what we found
        print(f"Found {len(resultTrees)} parse tree(s):")

        # only show first few to avoid spam
        for idx, tree in enumerate(resultTrees[:3]):
            print(f"\nParse Tree {idx + 1}:")
            print(tree)
            tree.pretty_print()

        if len(resultTrees) > 3:
            print(f"\n... and {len(resultTrees) - 3} more parse(s)")

        return resultTrees


def runCkyParsing():
    """Load the ATIS grammar and test CKY parsing on some sentences."""
    print("\n" + "=" * 60)
    print("TASK 2: CKY Parsing Algorithm")
    print("=" * 60)

    # try to load the ATIS grammar from NLTK
    print("\nLoading ATIS CFG grammar...")
    try:
        atisGrammar = nltk.data.load("grammars/large_grammars/atis.cfg")
        print(f"Grammar loaded with {len(atisGrammar.productions())} productions")
        print(f"Start symbol: {atisGrammar.start()}")
    except Exception as e:
        print(f"Error loading ATIS grammar: {e}")
        print("Please run: nltk.download('large_grammars')")
        return

    # create our parser
    print("\nConverting grammar to Chomsky Normal Form...")
    ckyParser = CkyParser(atisGrammar)
    print(f"CNF grammar has {len(ckyParser.grammar.productions())} productions")

    # these are the test sentences from the assignment
    testSentences = [
        "What is the cheapest one way flight from columbus to indianapolis",
        "Is there a flight from memphis to los angeles",
        "What aircraft is this",
        "Show american flights after twelve p.m. from miami to chicago"
    ]

    print("\n" + "=" * 60)
    print("Parsing Test Sentences")
    print("=" * 60)

    # run each one through the parser
    for sentence in testSentences:
        ckyParser.parseAndPrint(sentence)


# ============================================================================
# TASK 3a: Dependency Parsing with Stanford CoreNLP
# ============================================================================

def getDependencyParse(sentence):
    """
    Get a dependency parse using Stanford CoreNLP.

    Takes a sentence string and returns CoNLL-formatted output with:
    word, POS tag, head index, dependency relation

    Note: CoreNLP server must be running on port 9000!
    """
    from nltk.parse.corenlp import CoreNLPDependencyParser

    # try to connect to the CoreNLP server
    try:
        depParser = CoreNLPDependencyParser(url='http://localhost:9000')
    except Exception as connectionError:
        errorMessage = f"Error connecting to CoreNLP server: {connectionError}\n"
        errorMessage += "Please ensure CoreNLP server is running on port 9000.\n"
        errorMessage += 'Run: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000'
        return errorMessage

    # now try to actually parse the sentence
    try:
        parseResult = next(depParser.raw_parse(sentence))

        # convert to CoNLL format - 4 columns
        conllOutput = parseResult.to_conll(4)
        return conllOutput

    except StopIteration:
        return "No parse found for the sentence."
    except Exception as parseError:
        return f"Error parsing sentence: {parseError}"


def runDependencyParsing():
    """Demo the dependency parser with some example sentences."""
    print("\n" + "=" * 60)
    print("TASK 3a: Dependency Parsing with Stanford CoreNLP")
    print("=" * 60)

    # give clear setup instructions
    print("\n*** IMPORTANT: CoreNLP Server Setup Required ***")
    print("""
To use this dependency parser, you must:

1. Navigate to the Stanford CoreNLP folder:
   cd ASN5/stanford-corenlp-4.5.10

2. Start the CoreNLP server:
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

3. Then run this script again
""")

    # test sentences including ambiguous ones
    testSentences = [
        "The cat sat on the mat.",
        "Flying planes can be dangerous.",
        "Amid the chaos I saw her duck."
    ]

    print("\nTest Sentences for Dependency Parsing:")
    for idx, sentence in enumerate(testSentences, 1):
        print(f"{idx}. {sentence}")

    print("\nAttempting to parse (requires CoreNLP server)...")

    # try each one
    for sentence in testSentences:
        print(f"\nSentence: {sentence}")
        print("-" * 40)
        parseOutput = getDependencyParse(sentence)
        print(parseOutput)


# ============================================================================
# TASK 3b: Ambiguous Sentence Analysis
# ============================================================================

def analyzeAmbiguousSentences():
    """
    Analyze some famously ambiguous sentences.

    This is more of a written analysis showing how parsers
    can get confused by ambiguous structures.
    """
    print("\n" + "=" * 60)
    print("TASK 3b: Ambiguous Sentence Analysis")
    print("=" * 60)

    # this is the detailed analysis - pretty long but thorough
    analysisText = """
=== ANALYSIS OF AMBIGUOUS SENTENCES ===

1. "Flying planes can be dangerous"
   --------------------------------

   STRUCTURAL AMBIGUITY:
   This sentence has two possible interpretations:

   a) Reading 1 - "Flying planes" as a gerund phrase (action of flying):
      - "Flying" is a gerund (VBG) acting as the head of the subject
      - "planes" is the direct object of "flying"
      - Meaning: The act of piloting planes can be dangerous
      - Dependency: planes --(dobj)--> flying

   b) Reading 2 - "Flying planes" as noun phrase with adjective:
      - "Flying" is a participial adjective (JJ) modifying "planes"
      - "planes" is the head noun
      - Meaning: Planes that are currently flying can be dangerous
      - Dependency: flying --(amod)--> planes

   POTENTIAL PARSER ERRORS:
   - The parser may incorrectly assign "flying" as VBG when it should be JJ
   - The head-dependent relationship between "flying" and "planes" may be wrong
   - The parser typically chooses one interpretation, missing the ambiguity


2. "Amid the chaos I saw her duck"
   --------------------------------

   STRUCTURAL AMBIGUITY:
   This sentence has two possible interpretations:

   a) Reading 1 - "duck" as a noun (the bird):
      - "her" is a possessive determiner
      - "duck" is a noun (NN) being possessed
      - "saw" takes "duck" as direct object
      - Meaning: I saw the duck that belongs to her
      - Dependencies:
        * her --(poss)--> duck
        * duck --(dobj)--> saw

   b) Reading 2 - "duck" as a verb:
      - "her" is a direct object of "saw"
      - "duck" is a verb (VB) in a small clause construction
      - Meaning: I saw her perform the action of ducking
      - Dependencies:
        * her --(dobj)--> saw
        * duck --(xcomp)--> saw

   POTENTIAL PARSER ERRORS:
   - POS tag for "duck" may be incorrect (NN vs VB)
   - The relationship between "saw", "her", and "duck" depends on interpretation
   - "her" may be incorrectly labeled as poss instead of dobj or vice versa


3. ADDITIONAL AMBIGUOUS SENTENCE: "I made her duck orange sauce"
   --------------------------------------------------------------

   STRUCTURAL AMBIGUITY:
   Multiple interpretations exist:

   a) Reading 1 - Double object construction:
      - "her" is indirect object
      - "duck orange sauce" is direct object (a type of sauce)
      - Meaning: I prepared duck orange sauce for her

   b) Reading 2 - Causative construction with adjective:
      - "her duck" is object (her pet duck)
      - "orange sauce" describes the result
      - Meaning: I covered her duck with orange sauce

   c) Reading 3 - Complex causative:
      - "her" is object of "made"
      - "duck" is the verb she was made to do
      - "orange sauce" is an additional object/modifier
      - Very unusual interpretation

   POTENTIAL PARSER ERRORS:
   - POS ambiguity: "duck" (NN vs VB), "orange" (NN vs JJ)
   - Head assignment between "duck", "orange", and "sauce" is ambiguous
   - The parser will likely pick the most common structure, missing alternatives


=== KEY INSIGHTS ===

1. Syntactic ambiguity often arises from:
   - POS ambiguity (noun vs verb, adjective vs participle)
   - Attachment ambiguity (prepositional phrase, adjective attachment)
   - Structural ambiguity (complement vs adjunct)

2. Dependency parsers are typically trained on treebanks with single "gold" parses,
   so they cannot represent ambiguity - they must choose one interpretation.

3. Common error patterns:
   - Gerund/participle ambiguity (VBG vs JJ)
   - Noun/verb homographs (duck, fly, run, etc.)
   - Possessive vs objective case pronouns in certain contexts
"""

    print(analysisText)
    return analysisText


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all the parsing tasks in sequence."""
    print("=" * 60)
    print("Constituency and Dependency Parsing")
    print("=" * 60)

    # make sure we have the data we need
    print("\nChecking NLTK data...")
    downloadNltkData()

    # Task 1: build and show the constituency tree
    print("\n" + "=" * 60)
    print("Running Task 1: Constituency Tree Visualization")
    print("=" * 60)
    constituencyTree = createConstituencyTree()
    visualizeConstituencyTree(constituencyTree)

    # Task 2: CKY parsing with ATIS grammar
    print("\n" + "=" * 60)
    print("Running Task 2: CKY Parsing Algorithm")
    print("=" * 60)
    runCkyParsing()

    # Task 3a: dependency parsing (needs CoreNLP server)
    print("\n" + "=" * 60)
    print("Running Task 3a: Dependency Parsing")
    print("=" * 60)
    runDependencyParsing()

    # Task 3b: analysis of ambiguous sentences
    analyzeAmbiguousSentences()

    print("\n" + "=" * 60)
    print("All Tasks Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
