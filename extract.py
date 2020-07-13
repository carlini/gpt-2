import json
if __name__ == "__main__":
    from callcc import wrap
else:
    def wrap(fn):
        return fn
    import sys
    sys.path.append("src")
    import numpy as np
    import tensorflow as tf
    import difflib
    import heapq
    
    import sample
    import model
    import encoder
    


def proc(word):
    solution = [[],]
    
    for j in range(1,len(word)+1):
        subword = word[:j]
        
        best = None
        for i in range(1,min(len(subword)+1,maxlen)):
            if subword[-i:] in words:
                rest = solution[j-i]
                if rest is None:
                    continue
                option = rest + [words[subword[-i:]]]
                if best is None or len(option) < len(best) or (len(option) == len(best) and sum(option) < sum(best)):
                    best = option
        solution.append(best)
    return best

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]    
    
@wrap
def query(query, c):
    feed = np.array(enc.encode(query))

    out_toks = sess.run(tf_sample,
                           {ctx: feed[np.newaxis,:],
                            count: c})
    
    print(enc.decode(out_toks[0]) + "\n")

@wrap
def prob(q):
    feed = np.array(enc.encode(q))

    probs = sess.run(tf_next_probs,
                           {ctx: feed[np.newaxis,:],
                            count: 1})[0]
    best = np.argsort(-probs)

    o = ""
    for i in range(10):
        o += "p: " + str(probs[best[i]]) + " " + enc.decode([best[i]]) + "\n"

    return o

@wrap
def search(q, n, maxlen):
    prefix = enc.encode(q)
    #prefix = [0]

    seen = {}

    history = []
    queue = [(0, prefix)]

    while len(queue):
        if len(history) > int(maxlen): break
        if len(prefix) > int(n): break

        scores = []
        prefixes = []

        score, prefix = heapq.heappop(queue)
        scores.append(score)
        prefixes.append(prefix)
        history.append((score, enc.decode(prefix)))
        to_pop = []
        print("Progress: ", len(queue), "/", n)

        for i in range(len(queue)//10):
            s, p = queue[i]
            if len(p) == len(prefix):
                to_pop.append(i)
                scores.append(s)
                prefixes.append(p)
                history.append((s, enc.decode(p)))
            
            if len(prefixes) >= 10: break

        for i in to_pop[::-1]:
            queue.pop(i)
        
        feed = np.array(prefixes)
        probs = sess.run(tf_next_probs,
                         {ctx: feed,
                          count: 1})

        best = np.argsort(-probs,axis=1)

        for J in range(best.shape[0]):
            for i in range(40):
                p = probs[J,best[J,i]]
                if p < 1e-4:
                    break
                next_phrase = enc.encode(enc.decode(prefixes[J]+[best[J,i]]))
                if tuple(next_phrase) not in seen:
                    seen[tuple(next_phrase)] = True
                    # TODO this isn't quite right given the re-encoding
                    heapq.heappush(queue, (-np.log(p)+scores[J],
                                           next_phrase))
                    
    return history


@wrap
def batch(q, n, bs):
    feed = np.array(enc.encode(q))
    feed = np.stack([feed]*bs, axis=0)
    
    out_toks, q = sess.run((tf_sample, tf_context),
                           {ctx: feed,
                            count: int(n)})

    res = ""

    for toks in out_toks:
        redo = enc.encode(enc.decode(toks))
        print(difflib.SequenceMatcher(a=list(toks), b=list(redo)).get_opcodes())
        
        res += enc.decode(toks) + "\n"
        res += "="*40 + "\n\n"
        
    return res                                                                                                                                                                                

@wrap
def get_likelihood(q, ansi=True):
    feed = np.array(enc.encode(q))

    #print(feed)

    feed = np.stack([feed], axis=0)

    #print("LEN", feed.shape)
    
    probabilities = sess.run((tf_fwd_probs),
                             {ctx: feed,
                              count: feed.shape[1]})

    if ansi:
        color_order = ["\x1b[34m", "\x1b[35m", "\x1b[32m",
                       "\x1b[33m", "\x1b[31m"]
        res = ""

        for probs,toks in zip(probabilities,feed):
            phrase = enc.decode(toks, as_list=True)
            for p,e in zip(probs,phrase):
                res += color_order[int(p*5)] + e
        res += "\u001b[0m"
        
    else:
        res = '<body onload="run()">'
    
        for probs,toks in zip(probabilities,feed):
            phrase = enc.decode(toks, as_list=True)
            for p,e in zip(probs,phrase):
                res += '<span style="color: rgb(%f, %f, 0);">%s</span>'%(p*255,255-p*255,e)
    
        res += """qqq
    <script src="script.js">
    </script>
    </body>"""
    
    return res

@wrap
def init(which):
    global sess, graph, ctx, count, tf_fwd_probs, tf_sample, tf_probs, tf_context, enc, words, invwords, maxlen, tf_next_probs, enc

    sess = tf.Session()
    graph = tf.get_default_graph()

    hparams = model.default_hparams()

    with open("models/"+which+"/hparams.json") as f:
        hparams.override_from_dict(json.load(f))
    
    ctx = tf.placeholder(tf.int32, [None, None])
    count = tf.placeholder(tf.int32, [])

    tf_fwd_probs = sample.run_sequence(
        hparams=hparams,
        context=ctx,
        batch_size=1,
        length=count,
    )
    
    tf_sample, tf_probs, tf_context = sample.sample_sequence(
        hparams=hparams,
        length=count,
        context=ctx,
        temperature=1.0,
        top_k=10,
        top_p=0.5)

    tf_next_probs = sample.get_probs_next(
        hparams=hparams,
        length=count,
        context=ctx)
    

    sess.run(tf.global_variables_initializer())

    if which != "tiny":
        saver = tf.train.Saver()
        saver.restore(sess, "models/"+which+"/model.ckpt")

    enc = encoder.get_encoder(""+which+"", "models")

    words = json.loads(open("models/"+which+"/encoder.json").read())
    invwords = {v:k for k,v in words.items()}

    maxlen = max(map(len,words.keys()))

    return ""

def make_tree(data):
    class Tree:
        def __init__(self, val, children):
            self.val = val
            self.children = children
    
        def insert(self, path):
            if path == "": return
            for c in self.children:
                if c.val == path[0]:
                    c.insert(path[1:])
                    return
            t = Tree(path[0], [])
            t.insert(path[1:])
            self.children.append(t)
    
    
        def find(self, path):
            nodes = self.children
            for i,char in enumerate(path):
                found = [x for x in nodes if char == x.val]
                if len(found) > 0:
                    nodes = found[0].children
                else:
                    print("Abort", path[i:])
                    return path[:i]
            return path
                    
        def walk(self, t=""):
            node = self
            o = ""
            while len(node.children) == 1:
                o += node.val
                node = node.children[0]
    
            o += node.val
            if len(node.children) == 0:
                out = t+ o.replace("\n", "|").replace(" ", " ")
                print("%04d"%len(out), out)
                return
    
            [x.walk(t+o) for x in node.children]
        

        def walk2(self, t=""):
            node = self
            o = ""

            o = ""
            while len(node.children) == 1:
                o += node.val
                node = node.children[0]
            
            o += node.val
            print(t+ o)
            [x.walk2(t+" ") for x in node.children]
        
    tree = Tree('', [])

    for score,line in data:
        tree.insert(line)
    
    tree.walk("")

    
    
if __name__ == "__main__":
    # 1. load the model
    #init("124M")
    init("1558M")
    
    # 2. query the model on a single sequence
    query("This is a sample sentence", 30)

    # 3. get the likelihood of a sequence
    print(get_likelihood("This sentence is normal. Now I'm going to say something predictable. THIS SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND", ansi=True))
    
    open("/tmp/t", "w").write(json.dumps(search("https://", 100, 1000)))
    make_tree(json.load(open("/tmp/t")))
    
