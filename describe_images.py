print('importing PyTorch .. ', end='', flush=True)
import torch
from torch.nn.functional import interpolate
from transformers import AutoProcessor, AutoModelForCausalLM
print('done')

from PIL import Image
from glob import glob
from pathlib import Path
from tqdm import tqdm
from os import rename, path as ospath
# from pathlib import Path
from re import sub as regexSub
import argparse

cpu, gpu = torch.device("cpu"), torch.device("cuda")


imageExensions = [
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".dib", ".tiff", ".tif",
    ".eps", ".icns", ".ico", ".im", ".j2k", ".jp2", ".msp", ".pcx", ".pbm",
    ".pgm", ".ppm", ".pnm", ".sgi", ".spider", ".tga", ".xbm", ".blp", ".cur",
    ".dcx", ".dds", ".fli", ".flc", ".fpx", ".ftex", ".gbr", ".gd", ".imt",
    ".iptc", ".naa", ".mcidas", ".mic", ".mpo", ".pcd", ".pixar", ".psd",
    ".wal", ".xpm"
]

model_ID = "microsoft/Florence-2-large"


# load neural network and preprocessor for image description
def get_model():
        
    model = AutoModelForCausalLM.from_pretrained(
        model_ID,
        # torch_dtype = torch.float16 if device==gpu else torch.float32,
        torch_dtype = torch.float16,
        trust_remote_code=True)

    preproc = AutoProcessor.from_pretrained(
        model_ID,
        trust_remote_code=True)

    return model, preproc



def rename_at_path(oldPath, newName):
    newPath = ospath.join(
        ospath.dirname(oldPath), newName)
    if oldPath != newPath:
        rename(oldPath, newPath)



# returns an iterator over images in given folder path,
# sorted into batches 
def image_directory_iterator(globIterator, bsize = 16, 
                             verbose=False):


    files = globIterator
    n = len(files)
    failures = 0

    batch = []
    paths = []
    
    for i, imgPath in (pb := tqdm(enumerate(files), total=len(files))):
        
        pb.set_description_str(
            f"reading {i}/{len(files)}, {failures} skipped")
        

        

        try:
            pilImage = Image.open(imgPath).convert("RGBA")
            
            assert any(imgPath.lower().endswith(ext) for ext in imageExensions)
            
            # make sure it didnt accidently succeed in opening a non
            # image file as an image
            pilImage.verify()

            batch.append(pilImage)
            paths.append(imgPath)
        except Exception as e:
            if verbose:
                print(f"\nfailed to open file {imgPath}, {e}")
            failures += 1

        if len(batch) == bsize:
            yield n, failures, paths, batch, pb
            batch.clear()
            paths.clear()

    if len(batch) > 0:
        yield n, failures, paths, batch, pb

    # pb.set_description_str(f"{n}/{n}")



# make sure neural network output can be used to rename file
# purge characters that would raise error, and trucate length
def sanitize(text:str, maxLength):

    for notAllowed in ["<pad>", "The image shows a ", "."]:
        text = text.replace(notAllowed, "")   

    if text.lower().startswith("a "):
        text = text[2:]

    text = regexSub(r'[<>:"/\\|?*].,#~', '', text)
    
    return text[:maxLength]



# iterate over every image in given path and rename it using a 
# descriptive name
def describe_and_rename(model, formatter, path, bsize=2,
                        descriptionLengthLevel=1, parallelPaths=1, 
                        stochastic=False, maxTokens=1024,
                        hardPathCharLimit=255, forcedImageSize=None, 
                        searchSubFolders=True, verbose=False,
                        device=gpu):

    if verbose or True:
        print("counting files .. ", end="", flush=True)

    # glob requires ** for recursive search, * for regular
    extraPathFormat = '*' if searchSubFolders else ''
    formattedPath = \
        f"{path}//*{extraPathFormat}"\
        if len(path) > 0 else '*' + extraPathFormat
    # O(1) access and search dict for efficient description collision 
    # detection and resolution
    nameDict = {}
    globIterator = [
        _path for _path in 
        glob(formattedPath, recursive=searchSubFolders) 
        if not ospath.isdir(_path)]

    # pass existing names to dict to avoid rename errors
    for path in globIterator:
        rawName = ospath.splitext(ospath.basename(path))[0]
        nameDict[rawName] = 0


    model.to(device)

    with torch.no_grad():
        taskStr = \
            "<CAPTION>"          if descriptionLengthLevel==1 else \
            "<DETAILED_CAPTION>" if descriptionLengthLevel==2 else \
            "<MORE_DETAILED_CAPTION>"

    iterator = image_directory_iterator(globIterator, bsize, verbose)
    completed = renameFailed = readFailed = 0

    if verbose or True:
        print("done")

    for i, (n, failures, paths, batch, pb) in enumerate(iterator):

        # len(batch) != bsize means final iteration
        currentI = i*bsize if len(batch) == bsize else n
        pb.set_description_str(f"describing {currentI}/{n}")

        inputs = formatter(
            text=[taskStr for _ in batch],
            images=[img.convert("RGB") for img in batch],
            return_tensors = "pt",
            padding=True).to(device)

        pixels = inputs["pixel_values"].to(model.dtype)[:,:3,:,:]

        if forcedImageSize is not None:
            pixels = interpolate(pixels, forcedImageSize)

        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixels,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=maxTokens,
            num_beams=parallelPaths,
            do_sample=stochastic,
            early_stopping = parallelPaths > 1
        )

        decoded = formatter.batch_decode(
            out, skip_special_tokens=False)

        for text, imgPath in zip(decoded, paths):
            description = formatter.post_process_generation(
                text, task=taskStr)[taskStr]

            # current length of full path
            n = len(ospath.splitext(imgPath)[0])

            description = sanitize(description,
                                   hardPathCharLimit - n)

            currentName = ospath.splitext(ospath.basename(imgPath))[0]

            if currentName == description: 
                continue

            # make sure two files being described the exact same
            # does not overwrite anything
            if description in nameDict:
                nameDict[description] += 1
                # standard duplicate file naming
                description += f" ({nameDict[description]})"
            else:
                nameDict[description] = 0

            name = description + ospath.splitext(imgPath)[1]

            try:
                rename_at_path(imgPath, name)
            except Exception as e:
                if verbose:
                    print(f"\nfailed to rename file {imgPath}, {e}")
                renameFailed += 1

        completed += len(batch)
        readFailed = failures


    totalFailed = readFailed + renameFailed
    print(f"done! {completed} images described successfully, "
          f"{totalFailed} failed")

    if readFailed > 0:
        print(f"{readFailed} failed due to read errors")
    if renameFailed > 0:
        print(f"{renameFailed} failed due to renaming errors")

    if totalFailed > 0 and not verbose:
        print("run script with --verbose flag to see errors")


# prepare command line arguments
def prepare_arguments():
    parse = argparse.ArgumentParser(
        description=
        'Rename images in given folder with accurate descriptions!'    
    )

    parse.add_argument(
        'path', nargs='?', default=None,
        help='Directory containing images to describe')

    parse.add_argument(
        '-b', '--bsize', type=int, default=2,
        help='Working batch size - number of images to process at once, '
             'increasing this value speeds up the process but requires '
             'more memory (default: 2)')

    parse.add_argument(
        '-dc', '--descriptioncomplexity', type=int, default=1, choices=[1,2,3],
        help='Description detail level the model attempts to reach, '
             'larger values result in longer descriptions. ' 
             'Only increase past 1 if descriptions are generic'
             '(default: 1)')

    parse.add_argument(
        '-pp', '--parallelpaths', type=int, default=2,
        help='Model supports generating multiple description runs '
             'in parallel, and returning the highest scoring, '
             'proportionally increases running time, increase past 2 '
             'only if descriptions are of poor quality (default: 2)')

    parse.add_argument(
        '-s', '--stochastic', action='store_true',
        help='Enable random sampling')

    parse.add_argument(
        '-mt', '--maxtokens', type=int, default=1024,
        help='Max tokens to generate per image before halt')

    parse.add_argument(
        '-fs', '--forcedsize', type=int, default=None,
        help='Resize image to given resolution before passing '
             'to the description model, reduces memory usage '
             'at the cost of fine detail')

    parse.add_argument(
        '-r', '--searchsubfolders', default=False, action='store_true',
        help='Recursively search sub folders (default: False)')

    parse.add_argument(
        '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
        help='Device to use for inference, (default: cuda)')

    parse.add_argument(
        '-v', '--verbose', default=False, action='store_true',
        help='Print file read exceptions')

    parse.add_argument(
        '-f', '--fast', default=False, action='store_true',
        help='pre configured fast / low memory mode, may result in poor'
             'description quality (warning: overrides all arguments except '
             'recursive search, device, batch size and stochasticity)')


    return parse

# describe_and_rename(*get_model(), "imgs//")





# main command line call
if __name__ == '__main__':

    parse = prepare_arguments()
    args = parse.parse_args()

    if args.fast:
        parse.descriptioncomplexity = 1
        parse.parallelpaths = 1
        parse.maxtokens = 512
        parse.forcedsize = 128

    if args.path is None:
        print("no folder path given, press "
              "enter to run in current directory")
        input()
        args.path = ""

    if (not torch.cuda.is_available() and args.device=='cuda'):

        print('cuda passed as device argument but PyTorch cannot '
              'detect cuda device, make sure PyTorch was installed '
              'with cuda enabled, defaulting to cpu')


    device = gpu if args.device=='cuda' and torch.cuda.is_available() else cpu

    print(f"loading {model_ID} .. ", end="", flush=True)
    model = get_model()
    print("done")

    describe_and_rename(
        *model,
        args.path,
        args.bsize, args.descriptioncomplexity,
        args.parallelpaths, args.stochastic,
        args.maxtokens, 
        255, # windows caps path lengths at 260, 255 gives leeway 
        args.forcedsize, args.searchsubfolders, args.verbose,
        device)



