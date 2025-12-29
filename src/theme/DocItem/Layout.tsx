import React from 'react';
import OriginalDocItemLayout from '@theme-original/DocItem/Layout';
import TranslateRouteButton from '@site/src/components/LocaleRouter/TranslateButton';

export default function DocItemLayout(props) {
  return (
    <>
      <div className="margin-vert--lg">
        <TranslateRouteButton />
      </div>
      <OriginalDocItemLayout {...props} />
    </>
  );
}