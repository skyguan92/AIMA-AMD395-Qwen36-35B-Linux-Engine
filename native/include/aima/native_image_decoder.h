// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"
#include "aima/native_vl_processor.h"

namespace aima {

// Decodes the frozen PNG/JPEG/WebP image surface directly from admitted
// bytes. RGBA inputs use ImageMediaIO's request-effective RGB background.
NativeRgbFrame decode_native_image(const NativeMediaPayload& payload,
                                   const NativeMediaPolicy& policy);

}  // namespace aima
