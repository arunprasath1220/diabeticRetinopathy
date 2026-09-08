function [I, meta] = readFundus(fileName)
%READFUNDUS  Read a fundus image, DICOM or ordinary, and cap its working size.
%
%   [I, META] = dr.readFundus(FILE) returns an RGB uint8 image and whatever
%   acquisition metadata came with it.
%
%   Portable fundus cameras in a screening programme emit two quite different
%   things: consumer image files from phone-adapter rigs, and DICOM Ophthalmic
%   Photography objects from anything that talks to a hospital PACS. Both arrive
%   in the same district workflow, so both are read here rather than making the
%   operator convert. The Medical Imaging Toolbox path also carries the
%   acquisition metadata through to the report, which is what lets a result be
%   traced back to a device and a date.
%
%   The long edge is capped (see DR.CONFIG). Every threshold in the pipeline that
%   is expressed in pixels was measured at that scale, so this is not merely a
%   performance choice - running at native resolution would silently change what
%   counts as a microaneurysm.

    C = dr.Config();
    meta = struct("source", "image", "file", string(fileName));

    isDicom = false;
    try
        isDicom = isdicom(fileName);
    catch
        % isdicom throws on files it cannot open at all; treat as not DICOM and
        % let imread produce the real error message.
    end

    if isDicom
        info = dicominfo(fileName);
        raw = dicomread(info);
        meta.source = "dicom";
        meta.info = info;
        meta = copyIfPresent(meta, info, "Modality");
        meta = copyIfPresent(meta, info, "Manufacturer");
        meta = copyIfPresent(meta, info, "StudyDate");
        meta = copyIfPresent(meta, info, "Laterality");
    else
        raw = imread(fileName);
    end

    % Ophthalmic photography is stored as a single frame; a multi-frame object
    % means a video or a stereo pair, and taking frame 1 silently would hide that.
    if ndims(raw) == 4
        if size(raw,4) > 1
            warning("dr:readFundus:multiframe", ...
                "%d frames present; analysing the first only", size(raw,4));
        end
        raw = raw(:,:,:,1);
    end

    if size(raw,3) == 1
        raw = repmat(raw, 1, 1, 3);
    end
    if size(raw,3) > 3
        raw = raw(:,:,1:3);
    end
    I = im2uint8(raw);

    scale = min(1, C.workingMaxDim / max(size(I,1), size(I,2)));
    if scale < 1
        I = imresize(I, scale, "bilinear");
    end
    meta.workingSize = [size(I,1) size(I,2)];
end

function meta = copyIfPresent(meta, info, field)
    if isfield(info, field)
        meta.(field) = info.(field);
    end
end
