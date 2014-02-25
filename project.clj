(defproject rbm-clj "0.1.0-SNAPSHOT"
  :description "A basic Exponential Family Harmonium library in Clojure. Includes 'standard' Restricted Boltzmann Machine (RBM)."
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [org.clojure/math.numeric-tower "0.0.2"]
                 [net.mikera/core.matrix "0.20.0"]
                 [net.mikera/vectorz-clj "0.20.0"]
                 ;; the following are necessary only for the examples
                 [incanter "1.5.4"]
                 [gloss "0.2.2"]
                 [quil "1.6.0"]])
