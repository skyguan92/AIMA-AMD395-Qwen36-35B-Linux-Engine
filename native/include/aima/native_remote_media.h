// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aima/native_media.h"

#include <vector>

namespace aima {

// Internal remote-fetch boundary used by native_media.cpp. It is separated so
// local/data admission remains independently testable from the libcurl client.
NativeMediaTransport validate_native_remote_media_source(
    const NativeMediaPart& media, const NativeMediaPolicy& policy);
std::vector<unsigned char> fetch_native_remote_media(
    const NativeMediaPart& media, const NativeMediaPolicy& policy);

}  // namespace aima
