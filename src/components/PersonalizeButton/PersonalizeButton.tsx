import React, { useEffect, useState } from 'react';
import { useAuth } from '@site/src/auth';
import { AuthPersonalizeButton } from '@site/src/auth';
import styles from './styles.module.css';
import { useLocation } from '@docusaurus/router';

interface DocPersonalizeButtonProps {
  chapterTitle?: string;
  className?: string;
}

const DocPersonalizeButton: React.FC<DocPersonalizeButtonProps> = ({
  chapterTitle = 'Chapter Content',
  className = ''
}) => {
  const { user } = useAuth();
  const location = useLocation();
  const [actualTitle, setActualTitle] = useState(() => {
    // Initialize with document title if available
    if (typeof document !== 'undefined') {
      const docTitle = document.title.replace(' | Physical AI & Humanoid Robotics Platform', '');
      if (docTitle && docTitle !== 'Physical AI & Humanoid Robotics Platform') {
        return docTitle;
      }
    }
    return chapterTitle;
  });

  useEffect(() => {
    // Update title when location changes
    const updateTitle = () => {
      let newTitle = chapterTitle;

      // Get the actual document title from the document.title
      if (typeof document !== 'undefined') {
        const docTitle = document.title.replace(' | Physical AI & Humanoid Robotics Platform', '');
        if (docTitle && docTitle !== 'Physical AI & Humanoid Robotics Platform') {
          newTitle = docTitle;
        }
      }

      // Fallback to location-based detection if document title isn't available
      if (newTitle === 'Chapter Content' || !newTitle) {
        const pathParts = location.pathname.split('/').filter(part => part);
        if (pathParts.length > 0) {
          const lastPart = pathParts[pathParts.length - 1];
          newTitle = lastPart
            .replace(/-/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize first letter of each word
        }
      }

      setActualTitle(newTitle);
    };

    // Use setTimeout to ensure the document.title has updated after navigation
    const timer = setTimeout(updateTitle, 100);

    return () => clearTimeout(timer);
  }, [location.pathname, chapterTitle]);

  const chapterId = actualTitle.toLowerCase().replace(/\s+/g, '-');

  return (
    <div className={`${styles.personalizeSection} ${className}`}>
      <h3>Personalize this content for your level</h3>
      <AuthPersonalizeButton
        chapterId={chapterId}
        chapterTitle={actualTitle}
      />
    </div>
  );
};

export default DocPersonalizeButton;