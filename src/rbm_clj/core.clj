(ns rbm-clj.core
  (:require  [incanter.core :as incant]
             [incanter.stats :as istats]
             [clojure.core.matrix :as mtx]
             [clojure.core.matrix.operators :as m-op]
             [clojure.java.io :as jio]
             [clojure.math.numeric-tower :as cmath]))

(mtx/set-current-implementation :vectorz)
;; --------


;; ----  I/O functions ----

;;(gloss/defcodec lecun-mnist-image-codec
 ;; [:int32-be :int32-be :int32-be :int32-be (gloss/repeated :ubyte :prefix :none)])

(defn make-filepath
  [path filename]
  (str (if (= (last path) "/") path (str path "/")) filename))

 ;(defn load-binary-data
 ; [fname codec]
 ; (let [the-file (jio/file fname)
 ;       file-size (.length the-file)]
 ;   (with-open [inp (jio/input-stream the-file)]
 ;     (let [buf (byte-array file-size)]
 ;       (.read inp buf)
 ;       (gloss.io/decode codec (gloss.io/to-buf-seq buf))))))

(defn serialize
  "Print data structure to file"
  [data-struct #^String filename]
  (with-open [wrtr (jio/writer (java.io.File. filename))]
    (binding [*print-dup* true *out* wrtr] (prn data-struct))))

(defn serialize-matrix
  "Print matrix to file"
  [data-matrix #^String filename]
  (let [n-cols (mtx/column-count data-matrix)]
  (with-open [wrtr (jio/writer (java.io.File. filename))]
    (binding [*print-dup* true *out* wrtr] (prn (partition n-cols (mtx/eseq data-matrix)))))))

(defn deserialize [filename]
  (with-open [r (java.io.PushbackReader. (java.io.FileReader. filename))]
    (read r)))

(defn load-weights
  [path filename]
  (deserialize (make-filepath path filename)))

(defn save-weights
  [wts path filename]
  (let [filepath (make-filepath path filename)]
    (println (str "Saving weight matrix to " filepath))
    (serialize-matrix wts filepath)))

(defn load-rbm
  [path filename]
  (let [rbm (deserialize (make-filepath path filename))
        wts (:wts rbm)]
    (assoc rbm :wts (if-not (empty? wts) (mtx/matrix wts) wts))))

(defn save-rbm
  [rbm path filename]
  (let [filepath (make-filepath path filename)
        rbm-out (-> rbm
                    (assoc :vis0 nil)
                    (assoc :wts (mtx/to-nested-vectors (:wts rbm))))]
    (println (str "Saving RBM to " filepath))
    (serialize rbm-out filepath)))

;; -----------------------------------------------------
;; ------ Transfer functions and random samplers ------
(defn sample-bernoulli
  [p]
  (cond
   (= p 0.0) 0.0
   (= p 1.0) 1.0
   :else (istats/sample-binomial 1 :prob p)))

(defn sample-normal
  [mn]
  (+ mn (istats/sample-normal 1)))
  
(defn mtx-sigmoid
  [x]
  (mtx/div 1 (mtx/add 1 (mtx/exp (mtx/sub x)))))

(defn sigmoid
  [x]
  (/ 1 (inc (Math/exp (- x)))))

(defn categorical-probs
 [v]
 (mtx/div v (reduce + v)))
  

(defn mtx-sample-normal
  [mat]
  (let [samps (mtx/matrix (istats/sample-normal (apply * (mtx/shape mat))))]
    (mtx/add mat (mtx/reshape samps (mtx/shape mat)))))

(defn comp-mtx-sigmoid
  [m i j]
  (if (> j 0)
    (sigmoid (mtx/mget m i j))
    1))

(defn comp-mtx-sample-bernoulli
  [m i j]
  (let [val (double (mtx/mget m i j))]
    (cond
     (or (< val 0.0) (> val 1.0)) (throw (Exception. "what??"))
     :else (sample-bernoulli val))))

(defn get-compute-matrix-fn
  [has-bias? fn-seq]
  (if has-bias?
    (get-compute-matrix-fn false (cons (constantly 1) fn-seq))
    (fn [m i j] ((nth fn-seq j) (mtx/mget m i j)))))
;;----------------------------------------
;; ----- RBM definitions ------
(def default-rbm-params {:lr 0.001 :data-dir "" :name "defaultrbm"})

(def default-cycle-sequence [:up ; hid0-input
                             comp-mtx-sigmoid ; hid0-mean
                             comp-mtx-sample-bernoulli  ; hid0
                             :out ; conj hid0 onto output vector
                             :down ; vis1-input
                             comp-mtx-sigmoid ; vis1
                             :out ; conj vis1 onto output vector
                             :up ; hid1-input
                             comp-mtx-sigmoid ; hid1
                             :out])

(def default-rbm {:n-vis nil ; determined by data upon initialization
                  :n-hid nil ; if nil, defaults to n-vis
                  :vis-bias? true
                  :hid-bias? true
                  :params default-rbm-params
                  :cycle-seq default-cycle-sequence})

(def gauss-bernoulli-cycle-sequence [:up
                                     comp-mtx-sigmoid
                                     comp-mtx-sample-bernoulli
                                     :out
                                     :down
                                     :identity
                                     :out
                                     :up
                                     comp-mtx-sigmoid
                                     :out])

(def gauss-bernoulli-rbm {:n-vis nil
                          :n-hid nil
                          :vis-bias? true
                          :hid-bias? true
                          :params (assoc default-rbm-params :name "gaussbernoullirbm")
                          :cycle-seq gauss-bernoulli-cycle-sequence})

(defn get-mixed-gb-bernoulli-cycle-sequence
  [vis1-fn-seq]
  [:up
   comp-mtx-sigmoid
   comp-mtx-sample-bernoulli
   :out
   :down
   (get-compute-matrix-fn vis1-fn-seq)
   :out
   :up
   comp-mtx-sigmoid
   :out])

;(defn get-categorical-block-sum-mtx
;  [n k]
;  (letfn [(cycler [x d] (take (* n k) (drop d (cycle x))))]
;    (let [cyc-v [(repeat k 1) (repeat (- (* n k) k) 0)]
;          sov (map (partial cycler cyc-v) (* k (range n)))]
;      (mtx/matrix sov))))  
      
          
;; -------------------------------------

(defn merge-params
  [rbm-def params-map]
  (assoc-in rbm-def [:params] (merge (:params rbm-def) params-map)))

(defn get-init-wts
  "Return random weight matrix of specified size [#rows #cols]"
  [n-rows n-cols]
  (let [wts (mtx/matrix (istats/sample-normal (* n-rows n-cols) :sd 0.05))]
    (mtx/reshape wts [n-rows n-cols])))

(defn mtx-join-ones
  [data]
  (mtx/transpose (mtx/join (mtx/row-matrix (repeat (mtx/row-count data) 1)) (mtx/transpose data))))

(defn make-mask-matrix
  [data]
  (letfn [(comp-mask [m i j] (if (nil? (mtx/mget m i j)) 0 1))]
    (let [mask (mtx/matrix data)]
      (mtx/compute-matrix (mtx/shape mask) (partial comp-mask mask)))))

(defn- check-weights
  [wts n-rows n-cols]
  (let [new-wts (mtx/matrix wts)
        [wt-rows wt-cols] (mtx/shape new-wts)]
    (if-not (and (= wt-rows n-rows) (= wt-cols n-cols))
      (throw (Exception. (str "Weight matrix should be " n-rows "-by-" n-cols)))
      new-wts)))
  
(defn init-rbm
  ([rbm-def data] (init-rbm rbm-def data nil))
  ([rbm-def data in-wts]
     (let [n-vis (count (first data))
           n-hid (if (nil? (:n-hid rbm-def)) n-vis (:n-hid rbm-def))
           N (count data)
           n-wts-rows (if (:vis-bias? rbm-def) (inc n-vis) n-vis)
           n-wts-cols (if (:hid-bias? rbm-def) (inc n-hid) n-hid)
           wts (if in-wts (check-weights in-wts n-wts-rows n-wts-cols) (get-init-wts n-wts-rows n-wts-cols))
           vis0 (if (:vis-bias? rbm-def) (mtx-join-ones data) (mtx/matrix data))
           mask (if (nil? (:mask rbm-def))
                  nil
                  (if (:vis-bias? rbm-def)
                    (mtx-join-ones (make-mask-matrix (:mask rbm-def)))
                    (make-mask-matrix (:mask rbm-def))))]
       (assoc rbm-def :n-vis n-vis :n-hid n-hid :vis0 vis0 :wts wts :mask mask))))

;; -----------------
(defn rbm-put-new-data
  "Takes care of adding column of 1s to the data matrix, if vis layer has bias."
  [rbm data]
  (assert (= (:n-vis rbm) (count (first data))) ":n-vis not equal to # cols in data")
  (let [vis0 (if (:vis-bias? rbm) (mtx-join-ones data) (mtx/matrix data))]
    (assoc rbm :vis0 vis0)))

(defn rbm-put-new-weights
  [rbm wts]
  (let [new-wts (mtx/matrix wts)]
    (assert (= (mtx/shape new-wts) (mtx/shape (:wts rbm))) (str "wts shape should be " (mtx/shape (:wts rbm))))
    (assoc rbm :wts new-wts)))

(defn rbm-put-mask
  ([rbm tmp-mask] (rbm-put-mask rbm tmp-mask true))
  ([rbm tmp-mask make-mask-mtx?]
     (let [mask (if make-mask-mtx?
                  (if (:vis-bias? rbm) (mtx-join-ones (make-mask-matrix tmp-mask)) (make-mask-matrix tmp-mask))
                  tmp-mask)]
       (assoc rbm :mask mask))))
;; ----------------

(defn strip-bias-nodes
  "Strips the first column from a matrix. Used, e.g., for stripping the column of 1s from a reconstruction."
  [res-mtx]
  (mtx/matrix (mtx/submatrix res-mtx [nil [1 (dec (mtx/column-count res-mtx))]])))

(defn run-a-cycle
  ([cyc-seq vis0 wts] (run-a-cycle cyc-seq vis0 wts nil))
  ([cyc-seq vis0 wts mask]
  (loop [cycs cyc-seq out-vec [] prev-res vis0]
    (if (empty? cycs)
      out-vec
      (let [action (first cycs)
            res (cond
                 (= action :out) :out
                 (= action :up) (mtx/mmul prev-res wts)
                 (= action :down) (mtx/mmul prev-res (mtx/transpose wts))
                 (= action :identity) prev-res
                 (= action :sample-normal) (mtx-sample-normal prev-res)
                 (= action :mask) (mtx/mul prev-res mask)
                 :else (mtx/compute-matrix (mtx/shape prev-res) (partial action prev-res)))]
        (if (= res :out)
          (recur (rest cycs) (conj out-vec prev-res) prev-res)
          (recur (rest cycs) out-vec res)))))))

(defn rbm-run
  ([rbm data] (rbm-run rbm data (:cycle-seq rbm)))
  ([rbm data cyc-seq]
     (let [new-rbm (rbm-put-new-data rbm data)
           vis0 (:vis0 new-rbm)
           wts (:wts new-rbm)
           mask (:mask rbm)]
       (run-a-cycle cyc-seq vis0 wts mask))))

(defn clean-output-vec
  "This may be useful for semi-automatically cleaning up (e.g., removing bias nodes) outputs. There's probably a better way to implement this."
  [rbm label outvec]
  (let [out (cond
             (and (re-find #"vis" (str label)) (:vis-bias? rbm)) (strip-bias-nodes outvec)
             (and (re-find #"hid" (str label)) (:hid-bias? rbm)) (strip-bias-nodes outvec)
             :else outvec)]
    (mtx/to-nested-vectors out)))      
   
(defn rbm-cleanup-outputs
  [rbm cyc-seq-out-labels cyc-outputs]
  (assert (= (count cyc-seq-out-labels) (count cyc-outputs)) "unequal label output counts")
  (doall (map (partial clean-output-vec rbm) cyc-seq-out-labels cyc-outputs)))
    
(defn rbm-get-recon
  ([rbm data] (rbm-get-recon rbm data (:cycle-seq rbm)))
  ([rbm data cyc-seq]
     (let [new-rbm (rbm-put-new-data rbm data)
           vis0 (:vis0 new-rbm)
           wts (:wts new-rbm)
           vis1 (second (run-a-cycle cyc-seq vis0 wts (:mask rbm)))]
       (mtx/to-nested-vectors (if (:vis-bias? new-rbm) (strip-bias-nodes vis1) vis1)))))
  
(defn get-weights-update
  [v0 h0 v1 h1 lr]
  (letfn [(exp-op [v h N] (mtx/div (reduce mtx/add (map mtx/outer-product (mtx/rows v) (mtx/rows h))) N))]
  (let [N (mtx/row-count v0)
        vh0 (exp-op v0 h0 N)
        vh1 (exp-op v1 h1 N)]
    (mtx/mul lr (mtx/sub vh0 vh1)))))

(defn update-weights
  [params wts vis0 rbm-res-vec]
  (let [dw (apply (partial get-weights-update vis0) (conj rbm-res-vec (:lr params)))]
    (mtx/add wts dw)))

(defn get-recon-error
  [data recon]
  (/ (reduce + (map (fn [d dhat] (cmath/abs (- dhat d))) (mtx/as-vector data) (mtx/as-vector recon))) (mtx/row-count data)))

(defn check-recon-error
  [rbm wts]
  (let [rbm-res (run-a-cycle (:cycle-seq rbm) (:vis0 rbm) wts (:mask rbm))]
    (println (str "recon err: " (get-recon-error (:vis0 rbm) (second rbm-res))))))

(defn train-rbm
  [rbm n-iters]
  (let [cycle-seq (:cycle-seq rbm)
        vis0 (:vis0 rbm)
        params (:params rbm)
        mask (:mask rbm)
        err-every-n (max 1 (inc (int (/ n-iters 5))))
        print-iters-every-n (max 1 (int (/ err-every-n 10)))]
  (loop [i n-iters wts (mtx/mutable (:wts rbm))]
    (if (= i 0)
      (do (check-recon-error rbm wts)
          (mtx/matrix wts))
      (let [rbm-res (run-a-cycle cycle-seq vis0 wts mask)
            dw (get-weights-update vis0 (first rbm-res) (second rbm-res) (nth rbm-res 2) (:lr params))]
        (mtx/add! wts dw)
        (if (= 0 (rem i print-iters-every-n)) (println (str "iters left: " (dec i))))
        (if (= 0 (rem i err-every-n))
          (println (str "  recon err: " (get-recon-error vis0 (second rbm-res)))))
        (recur (dec i) wts))))))


