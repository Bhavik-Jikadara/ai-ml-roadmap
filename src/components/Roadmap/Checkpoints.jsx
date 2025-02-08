import { useState, } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  ExternalLink, ChevronRight, Clock,
} from 'lucide-react';
import PropTypes from 'prop-types';

const Checkpoint = ({ title, items, index, isVisible }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`transform transition-all duration-700 my-8 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0'
      }`}>
      <Card className="mb-8 bg-gradient-to-br from-gray-800 to-gray-900 border-gray-700 hover:shadow-xl transition-shadow mx-4 md:mx-8">
        <CardContent className="p-6 md:p-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <Badge className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-3 py-1">
                Step {index + 1}
              </Badge>
              <h2 className="text-xl font-bold text-white">{title}</h2>
            </div>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <ChevronRight className={`w-6 h-6 transform transition-transform ${isExpanded ? 'rotate-90' : ''
                }`} />
            </button>
          </div>

          <div className={`space-y-3 transition-all duration-300 ${isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0 overflow-hidden'
            }`}>
            {Array.isArray(items) && items.map((item, idx) => (
              <div key={idx} className="flex items-center space-x-3 text-gray-300 hover:text-white transition-colors">
                {typeof item === 'string' ? (
                  <div className="flex items-center space-x-2">
                    <Clock className="w-4 h-4 text-gray-500" />
                    <span>{item}</span>
                  </div>
                ) : (
                  // Avoid nesting links
                  <div className="flex items-center space-x-2">
                    <ExternalLink className="w-4 h-4 text-gray-500" />
                    <a
                      href={item.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-blue-400"
                    >
                      {item.text}
                    </a>
                  </div>
                )}

              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

Checkpoint.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.array.isRequired,
  index: PropTypes.number.isRequired,
  isVisible: PropTypes.bool.isRequired,
};

export default Checkpoint;