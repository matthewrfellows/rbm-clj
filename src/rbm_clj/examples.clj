(ns rbm-clj.examples
  (:require [rbm-clj.core :as rbm]
           ;; [clojure.core.matrix :as mtx]
            [clojure.string]
            [clojure.java.io :as jio]
            [gloss.core :as gloss]
            [gloss.io]
            [quil.core :as quil]
           ;; [quil.helpers.seqs :as quilhs]
           ;; [quil.helpers.calc :as quilhc]
           ;; [incanter.stats :as istats]
            [clojure.math.numeric-tower :as cmath]))


;; ------------
;; The image data needed to run some of the examples can be found
;; here: http://yann.lecun.com/exdb/mnist/.
;; Download and uncompress the file called t10k-images.idx3-ubyte.gz.
;; Set the path in test-file-name, below to point to this file.

;; Set the following to the appropriate path
(def test-file-name "path_to_image_data/t10k-images.idx3-ubyte")

;; Some of the examples save data to disk. 
;; Set :data-dir to an appropriate path for saving.
(def rbm-params {:lr 0.001 :data-dir "path_to_output_directory"})


;;(def local-default-rbm (rbm/merge-params rbm/default-rbm rbm-params))

;; ------------------------------------------
(gloss/defcodec image-codec [:int32-be :int32-be :int32-be :int32-be (gloss/repeated :ubyte :prefix :none)])

(defn load-binary-data
  [fname codec]
  (let [the-file (jio/file fname)
        file-size (.length the-file)]
    (with-open [inp (jio/input-stream the-file)]
      (let [buf (byte-array file-size)]
        (.read inp buf)
        (gloss.io/decode codec (gloss.io/to-buf-seq buf))))))

(defn parse-iters-from-filename
  [fname]
  (clojure.string/split (re-find #"_\d*iters\.edn" fname) #"_|iters\.edn"))

(defn threshold
  [img-vec]
  (map (fn [x] (int (cmath/round (/ x 255)))) img-vec))

(defn just-scale
  [img-vec]
  (map #(double (/ % 79)) img-vec))

(defn get-images
  [fname]
  (println fname)
  (let [test-data (load-binary-data fname image-codec)
        n-images (second test-data)
        n-rows (nth test-data 2)
        n-cols (nth test-data 3)
        images (partition (* n-rows n-cols) (nth test-data 4))]
    (hash-map :n-rows n-rows :n-cols n-cols :images images)))

(defn get-image
  ([i img-map] (get-image i img-map true))
  ([i img-map threshold?]
     (let [img (->> (nth (:images img-map) i)
                    (partition (:n-cols img-map)))]
       (if threshold?
         (map threshold img)
         (map just-scale img)))))

(defn vec-to-image
  ([n-cols v] (vec-to-image n-cols v true))
  ([n-cols v threshold?]
     (if threshold?
       (map threshold (partition n-cols (map #(* 255 %) v)))
       (map just-scale (partition n-cols (map #(* 79 %) v))))))

(defn get-dataset
  ([img-map] (get-dataset img-map (count (:images img-map)) true))
  ([img-map N] (get-dataset img-map 0 N true))
  ([img-map starti N] (get-dataset img-map starti N true))
  ([img-map starti N threshold?]
     (let [imgs (:images img-map)]
       (take N (drop starti (if threshold? (map threshold imgs) (map just-scale imgs)))))))

#_(defn get-dataset-no-thresh
  [img-map starti N]
  (take N (drop starti (:images img-map))))

(def sq-side 25)

(defn draw-point
  [x y]
  (let [len sq-side]
    (quil/rect x y len len)))

(defn draw-squares
  [img]
  (let [ny (count img)
        nx (count (first img))]
  (dorun
   (for [j (range nx)
         i (range ny)
         :let [y (* i sq-side) x (* j sq-side)]]
     (do
       (quil/fill (* 255 (nth (nth img i) j)))
       (draw-point x y))))))

(defn setup [img]
  (quil/smooth)
  (quil/background 255)
  (quil/stroke 5)
  (quil/stroke-weight 1))

(defn draw-image
  ([img] (draw-image img "MNIST Digit"))
  ([img title-str]
     (quil/sketch :title title-str
                  :setup (fn [] (setup img))
                  :draw (fn [] (draw-squares img))
                  :size [(* sq-side (count (first img))) (* sq-side (count img))])))

;; --------------------------

(defn train-new-rbm
  [rbm-def dataset n-iters save-wts?]
  (let [rbm (rbm/init-rbm rbm-def dataset)
        trained-wts (rbm/train-rbm rbm n-iters)
        rbm-name (:name (:params rbm))
        out-dir (:data-dir (:params rbm))
        filename (str "trained_" rbm-name "_wts_" n-iters "iters.edn")]
    (if save-wts?
      (do
        (println "saving to" out-dir)
        (rbm/save-weights trained-wts out-dir filename)
        [rbm filename])
      [rbm trained-wts])))

(defn train-new-gb-rbm
  [starti N n-iters save-wts?]
  (let [ds (get-dataset (get-images test-file-name) starti N false)
        rbm-def (rbm/merge-params rbm/gauss-bernoulli-rbm rbm-params)
        [rbm trained-wts-or-filename] (train-new-rbm rbm-def ds n-iters save-wts?)]
       [rbm trained-wts-or-filename]))

(defn train-new-default-rbm
  ([starti N n-iters] (train-new-default-rbm starti N n-iters false))
  ([starti N n-iters save-wts?]
     (let [ds (get-dataset (get-images test-file-name) starti N)
           rbm-def (rbm/merge-params rbm/default-rbm rbm-params)
           [rbm trained-wts-or-filename] (train-new-rbm rbm-def ds n-iters save-wts?)]
       [rbm trained-wts-or-filename])))
           
(defn continue-training-rbm
  [rbm-def wts-filename dataset n-iters]
  (let [data-dir (:data-dir (:params rbm-def))
        wts (rbm/load-weights data-dir wts-filename)
        rbm-name (:name (:params rbm-def))
        rbm (rbm/init-rbm rbm-def dataset wts)
        trained-wts (rbm/train-rbm rbm n-iters)
        n-prev-iters (try (Integer. (second (parse-iters-from-filename wts-filename)))
                          (catch Exception e
                            nil))
        out-filename (str "trained_" rbm-name "_wts_" (+ n-iters n-prev-iters) ".edn")]
    (rbm/save-weights trained-wts data-dir out-filename)
    out-filename))

;; -----------------------------------------------------------------------------
;; ------- Examples below ---------

(defn example0
  "Train a new RBM on a small dataset. Save the trained weights to a file."
  ([] (example0 5))
  ([n-iters]
     (let [ds (get-dataset (get-images test-file-name) 0 20)
           rbm-def (rbm/merge-params rbm/default-rbm rbm-params)
           [rbm trained-wts-filename] (train-new-rbm rbm-def ds n-iters true)]
       (println (dissoc rbm :wts :vis0))
       (let [ds2 (get-dataset (get-images test-file-name) 20 50)
             new-wts-filename (continue-training-rbm rbm trained-wts-filename ds2 n-iters)]
         (println new-wts-filename)))))


(defn example1
  "Grab a small dataset, briefly train a new RBM, save it, reload it. Then put in a data sample and return its reconstruction."
  ([] (example1 5))
  ([n-iters]
     (let [ds (get-dataset (get-images test-file-name) 0 20)
           rbm-def (rbm/merge-params rbm/default-rbm rbm-params)
           data-dir (:data-dir (:params rbm-def))
           [rbm trained-wts] (train-new-rbm rbm-def ds n-iters false)]
       (rbm/save-rbm (rbm/rbm-put-new-weights rbm trained-wts) data-dir "temprbm_.edn")
       (let [loaded-rbm (rbm/load-rbm data-dir "temprbm_.edn")]
         (rbm/rbm-get-recon loaded-rbm (get-dataset (get-images test-file-name) 0 1))))))


(defn example2
  "Train a new RBM, draw a digit image, and its reconstruction."
  ([] (example2 5 50))
  ([n-iters n-datums]
     (let [draw-nth 0
           img-map (get-images test-file-name)
           ds (get-dataset img-map 0 n-datums)
           rbm-def (rbm/merge-params rbm/default-rbm rbm-params)
           data-dir (:data-dir (:params rbm-def))
           [rbm trained-wts] (train-new-rbm rbm-def ds n-iters false)
           trained-rbm (assoc rbm :wts trained-wts)
           recon (rbm/rbm-get-recon trained-rbm (get-dataset img-map draw-nth 1))
           recon-img (vec-to-image (:n-cols img-map) (first recon))]
       (draw-image (get-image draw-nth img-map))
       (draw-image recon-img (str "recon, after" n-iters " training iterations")))))

(defn example3
  "Train a new Exponential family Harmonium (Gaussian vis layer, Bernoulli hidden layer), draw an unthresholded digit image, and its reconstruction."
  ([] (example3 5 50))
  ([n-iters n-datums]
     (let [draw-nth 0
           img-map (get-images test-file-name)
           ds (get-dataset img-map 0 n-datums false)
           rbm-def (rbm/merge-params rbm/gauss-bernoulli-rbm rbm-params)
           data-dir (:data-dir (:params rbm-def))
           [rbm trained-wts] (train-new-rbm rbm-def ds n-iters false)
           trained-rbm (assoc rbm :wts trained-wts)
           recon (rbm/rbm-get-recon trained-rbm (get-dataset img-map draw-nth 1 false))
           recon-img (vec-to-image (:n-cols img-map) (first recon) false)]
       (draw-image (get-image draw-nth img-map false))
       (draw-image recon-img (str "recon, after" n-iters " training iterations")))))
       
;; --------------------------------------------
