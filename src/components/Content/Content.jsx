import { useState } from 'react';
import { ChevronRight, ExternalLink, ArrowRight } from 'lucide-react';
import Medium from '../../assets/medium.png';
import PropTypes from 'prop-types';

export const Header = () => (
  <div className="relative overflow-hidden mb-16 mt-24 rounded-2xl">
    <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 blur-3xl" />
    <div className="relative bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 p-12 md:p-16 rounded-2xl border border-gray-800/50 shadow-2xl">
      <div className="max-w-4xl mx-auto relative">
        {/* Animated background elements */}
        <div className="absolute inset-0">
          <div className="absolute top-0 right-0 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-0 left-0 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-700" />
        </div>

        {/* Content */}
        <div className="relative z-10 space-y-8">
          <div className="space-y-4">
            <div className="flex items-center space-x-3 mb-6">
              <span className="px-4 py-1 bg-blue-600/20 text-blue-400 rounded-full text-sm font-medium">
                2025 Edition
              </span>
              <span className="px-4 py-1 bg-purple-600/20 text-purple-400 rounded-full text-sm font-medium">
                Updated Weekly
              </span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 animate-fade-in">
              AI/ML Roadmap 2025
            </h1>
            <p className="text-xl text-gray-300 leading-relaxed max-w-2xl animate-slide-up">
              Master Artificial Intelligence and Machine Learning with our comprehensive learning path.
              Access curated resources and join a community of learners.
            </p>
          </div>

          <div className="flex flex-wrap gap-6">
            <a
              href="#resources"
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-xl font-medium transition-all transform hover:scale-105"
            >
              <span>Get Started</span>
              <ArrowRight className="w-4 h-4" />
            </a>
            <a
              href="#community"
              className="flex items-center space-x-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-xl font-medium transition-all border border-gray-700"
            >
              <img src={Medium} alt="Medium" className="w-6 h-6" />
                            <a href="https://bhavikjikadara.medium.com" target="_blank" rel="noopener noreferrer"><span>Follow on Medium</span></a>
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
);

export const Checkpoint = ({ title, items, index, isVisible }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div 
      className={`transform transition-all duration-700 ${
        isVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
      }`}
    >
      <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-xl p-6 mb-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-800">
        {/* Header */}
        <div 
          className="flex items-center justify-between mb-6 cursor-pointer group"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center transform transition-transform group-hover:scale-110">
                <span className="text-white font-bold text-lg">{index + 1}</span>
              </div>
              {index < items.length - 1 && (
                <div className="absolute top-12 left-1/2 w-0.5 h-8 bg-gradient-to-b from-blue-600/50 to-transparent" />
              )}
            </div>
            <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              {title}
            </h2>
          </div>
          <ChevronRight 
            className={`w-6 h-6 text-gray-400 transform transition-transform duration-300 group-hover:text-blue-400 ${
              isExpanded ? 'rotate-90' : ''
            }`}
          />
        </div>
        
        {/* Content */}
        <div className={`space-y-3 transition-all duration-500 ${
          isExpanded ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
        }`}>
          {items.map((item, idx) => (
            <div 
              key={idx}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10 rounded-xl blur group-hover:opacity-100 opacity-0 transition-opacity" />
              <div className="relative bg-gray-800/50 backdrop-blur-sm p-4 rounded-xl hover:bg-gray-700/50 transition-all border border-gray-700/50 group-hover:border-blue-500/50">
                {item.href ? (
                  <a
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between w-full group"
                  >
                    <span className="text-gray-200 group-hover:text-blue-400 transition-colors">
                      {item.text || item}
                    </span>
                    <ExternalLink className="w-4 h-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-all transform group-hover:translate-x-1" />
                  </a>
                ) : (
                  <span className="text-gray-200">{item.text || item}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

Checkpoint.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.oneOfType([
      PropTypes.string,
      PropTypes.shape({
        text: PropTypes.string,
        href: PropTypes.string
      })
    ])
  ).isRequired,
  index: PropTypes.number.isRequired,
  isVisible: PropTypes.bool.isRequired
};

export default { Header, Checkpoint };