import { useState, useEffect } from 'react';
import { AlertDialog, AlertDialogContent, AlertDialogHeader, AlertDialogTitle, AlertDialogDescription, AlertDialogFooter, AlertDialogAction } from '../ui/alert-dialog';
import { Navbar } from '../Header/Header';
import { Footer } from '../Footer/Footer';
import { Header, Checkpoint } from '../Content/Content';
import { BookOpen, ExternalLink } from 'lucide-react';
import { NeuralBackground } from '../ui/NeuralBackground';
import PropTypes from 'prop-types';

const WelcomeDialog = ({ open, onOpenChange }) => (
  <AlertDialog open={open} onOpenChange={onOpenChange}>
    <AlertDialogContent className="bg-gray-900 text-white border border-gray-800  ">
      <AlertDialogHeader>
        <AlertDialogTitle className="text-2xl font-bold text-white">
          Welcome to AI/ML Roadmap 2024! 🚀
        </AlertDialogTitle>
        <AlertDialogDescription className="text-gray-300">
          <p className="mb-4">
            Start your journey into AI and Machine Learning with our comprehensive guide.
            We have organized everything you need to know into a clear, step-by-step path.
          </p>
          <ul className="space-y-2">
            <li className="flex items-center">
              <BookOpen className="w-4 h-4 mr-2 text-blue-500" />
              Access curated resources
            </li>
            <li className="flex items-center">
              <ExternalLink className="w-4 h-4 mr-2 text-green-500" />
              Join our learning community
            </li>
          </ul>
        </AlertDialogDescription>
      </AlertDialogHeader>
      <AlertDialogFooter>
        <AlertDialogAction className="bg-blue-600 hover:bg-blue-700 text-white">
          Let&apos;s Begin!
        </AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  </AlertDialog>
);

WelcomeDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onOpenChange: PropTypes.func.isRequired,
};

const RoadmapApp = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [visibleCheckpoints, setVisibleCheckpoints] = useState(new Set());

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setVisibleCheckpoints(prev => new Set([...prev, parseInt(entry.target.dataset.index)]));
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('.checkpoint-container').forEach(checkpoint => {
      observer.observe(checkpoint);
    });

    return () => observer.disconnect();
  }, []);

  // Your checkpoints data here...
   const checkpoints = [
    {
      title: "Fundamentals of Programming",
      items: [
        { text: "Python Programming", href: "https://youtu.be/rfscVS0vtbw" },
        { text: "R Programming", href: "https://youtu.be/_V8eKsto3Ug" },
        { text: "Algorithms and Data Structures", href: "https://youtu.be/2ZLl8GAk1X4" },
        { text: "Problem Solving Techniques", href: "https://youtu.be/oBt53YbR9Kk" },
        { text: "OOP Concepts", href: "https://youtu.be/Ej_02ICOIgs" }
      ]
    },
    {
      title: "Mathematics for AI/ML",
      items: [
        { text: "Calculus", href: "https://math.mit.edu/~djk/calculus_beginners/" },
        { text: "Linear Algebra", href: "https://immersivemath.com/ila/learnmore.html" },
        { text: "Probability and Statistics", href: "https://www.khanacademy.org/math/statistics-probability" },
        { text: "Differential Equations", href: "https://www.khanacademy.org/math/differential-equations" },
        { text: "Discrete Mathematics", href: "https://youtube.com/playlist?list=PLHXZ9OQGMqxersk8fUxiUMSIx0DBqsKZS" },
        { text: "Optimization Technologies", href: "https://www.youtube.com/playlist?list=PLLtQL9wSL16ioUvHckGCkoWq_CIvyUI0p" }
      ]
    },
    {
      title: "Basics of AI/ML",
      items: [
        {
          text: "Introductions to Supervised and Unsupervised Learning in Machine Learning, Neural Networks, Deep Learning, and Reinforcement Learning",
          href: "https://youtu.be/ukzFI9rgwfU"
        }
      ]
    },
    {
      title: "Data Skills for AI/ML",
      items: [
        { text: "Data Collection", href: "https://labelyourdata.com/articles/data-collection-methods-AI" },
        { text: "Data Cleaning and Processing", href: "https://monkeylearn.com/blog/data-cleaning-steps/" },
        { text: "Feature Engineering", href: "https://builtin.com/articles/feature-engineering" },
        { text: "Exploratory Data Analysis", href: "https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15" },
        { text: "Data Visualization Technologies", href: "https://www.datacamp.com/blog/data-visualization-techniques" },
        { text: "Use of Libraries like Pandas, Numpy Matplotlib, Seaborn", href: "https://www.kaggle.com/discussions/getting-started/251992" }
      ]
    },
    {
      title: "Machine Learning",
      items: [
        { text: "Linear Regression, Logistic Regression", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" },
        { text: "Decision Trees and Random Forests", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" },
        { text: "Support Vector Machines", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" },
        { text: "K-Nearest Neighbours", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" },
        { text: "Naive Bayes", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" },
        { text: "XGBoost, LightGBM, CatBoost", href: "https://www.youtube.com/watch?v=JxgmHe2NyeY" }
      ]
    },
    {
      title: "Deep Learning",
      items: [
        { text: "ANN and CNN", href: "https://www.youtube.com/watch?v=d2kxUVwWWwU" },
        { text: "RNN and LSTM", href: "https://www.theaidream.com/post/introduction-to-rnn-and-lstm" },
        { text: "GANs", href: "https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/" },
        { text: "Transformer Models", href: "https://blogs.nvidia.com/blog/what-is-a-transformer-model/" },
        { text: "Deep Learning Libraries (Tensorflow, PyTorch, Keras)", href: "https://www.datacamp.com/tutorial/pytorch-vs-tensorflow-vs-keras" }
      ]
    },
    {
      title: "Natural Language Processing",
      items: [
        { text: "Text Preprocessing Techniques", href: "https://www.youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn" },
        { text: "Word Embeddings", href: "https://www.youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn" },
        { text: "Bag of Words, TF-IDF", href: "https://www.youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn" },
        { text: "LSTMs, GRU", href: "https://www.youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn" },
        { text: "Transformers and BERT", href: "https://youtu.be/7kLi8u2dJz0" }
      ]
    },
    {
      title: "Computer Vision",
      items: [
        { text: "Image Processing Techniques", href: "https://youtu.be/IA3WxTTPXqQ" },
        { text: "Convolutional Neural Networks", href: "https://youtu.be/IA3WxTTPXqQ" },
        { text: "Object Detection Algorithms", href: "https://youtu.be/IA3WxTTPXqQ" },
        { text: "Image Segmentation", href: "https://youtu.be/IA3WxTTPXqQ" },
        { text: "Facial Recognition Techniques", href: "https://youtu.be/IA3WxTTPXqQ" }
      ]
    },
    {
      title: "Reinforcement Learning",
      items: [
        { text: "Bellman Equation", href: "https://youtu.be/14BfO5lMiuk" },
        { text: "Q-Learning", href: "https://youtu.be/0iqz4tcKN58" },
        { text: "SARSA (State-Action-Reward-State-Action)", href: "https://www.geeksforgeeks.org/sarsa-reinforcement-learning/" },
        { text: "Deep Q-Network", href: "https://www.tensorflow.org/agents/tutorials/0_intro_rl" },
        { text: "Policy Gradient Methods", href: "https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d" },
        { text: "Monte Carlo Methods", href: "https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/" }
      ]
    },
    {
      title: "Tools and Libraries",
      items: [
        "Scipy, Scikit-Learn",
        "Keras, TensorFlow, PyTorch",
        "OpenCV",
        "Matplotlib, Seaborn, Plotly",
        "Pandas, Numpy",
        "Jupyter Notebook, Jupyter Lab"
      ]
    },
    {
      title: "Build AI/ML Applications",
      items: [
        "End-to-end Model Development",
        "Application Integration(Web, Mobile)",
        "Using ML Cloud Platforms(AWS, Azure, GCP)",
        "Using AI API Services",
        "Deployment and Scaling of Models",
        "Model Optimization Techniques"
      ]
    },
    {
      title: "Knowledge on Recent Trends and Advancements",
      items: [
        "Quantum Computing",
        "Federated Learning",
        "AI Ethics and Fairness",
        "AutoML and Neural Architecture Search",
        "Explainable AI",
        "AI in Edge devices"
      ]
    },
    {
      title: "The Super Duper NLP Repo",
      items: [
        { text: "The Super Duper NLP Repo &apos;", href: "https://notebooks.quantumstat.com/" }
      ]
    }
  ];
  
  return (
    <div className="min-h-screen relative overflow-hidden bg-gray-900">
      <NeuralBackground />
      <Navbar />
      <WelcomeDialog open={showWelcome} onOpenChange={setShowWelcome} />
      
      <main className="max-w-4xl mx-auto px-4 py-8 relative">
          <Header />
        {checkpoints.map((checkpoint, index) => (
          <div key={index} className="checkpoint-container" data-index={index}>
            <Checkpoint 
              index={index}
              title={checkpoint.title}
              items={checkpoint.items}
              isVisible={visibleCheckpoints.has(index)}
            />
          </div>
        ))}
      </main>

      <Footer />
    </div>
  );
};


export default RoadmapApp;
